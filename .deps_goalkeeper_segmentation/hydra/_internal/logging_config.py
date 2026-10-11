# SPDX-FileCopyrightText: Contributors to Hydra
# SPDX-License-Identifier: MIT

import logging.config
import logging.handlers
import sys
import warnings
from textwrap import dedent
from typing import Any, Callable, Dict, Tuple

from hydra._internal.target_policy import (
    _EXPECTED_TARGET_POLICY_DIGEST,
    _authorize_discovery_path,
    _authorize_resolved_target_identity,
    _authorize_target_invocation,
    _authorize_target_name,
    _get_os_alias_target,
    _get_resolved_target_name_for_check,
    _mediate_target_result,
    _reject_protected_result,
    _target_policy_context,
    _validated_target_policy,
)
from hydra.errors import InstantiationException


class HydraDictConfigurator(logging.config.DictConfigurator):
    """Apply Hydra target authorization to Python logging configuration."""

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)
        self._target_policy = _validated_target_policy(_EXPECTED_TARGET_POLICY_DIGEST)
        self._resolved_targets: Dict[str, Any] = {}
        self._resolved_target_sources: Dict[int, str] = {}

    def configure(self) -> None:
        with _target_policy_context(self._target_policy):
            super().configure()

    def _authorize_callable(self, target: Any, resolved_from: str) -> str:
        if not callable(target):
            return ""
        if resolved_from:
            return _authorize_resolved_target_identity(
                target, resolved_from, "hydra.logging"
            )
        target_name = _get_os_alias_target(_get_resolved_target_name_for_check(target))
        _authorize_target_name(target_name, target_name, "hydra.logging")
        return target_name

    def _invoke_authorized_callable(
        self,
        target: Callable[..., Any],
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
        resolved_from: str,
    ) -> Any:
        effective_target, effective_args, effective_kwargs = (
            _authorize_target_invocation(
                target,
                args,
                kwargs,
                "hydra.logging",
            )
        )
        discovery_path = _authorize_discovery_path(
            effective_target,
            effective_args,
            effective_kwargs,
            "hydra.logging",
        )
        result = target(*args, **kwargs)
        return _mediate_target_result(
            result,
            discovery_path or resolved_from,
            "hydra.logging",
            resolved_from_is_alias=discovery_path is not None,
        )

    def resolve(self, s: str) -> Any:
        if s in self._resolved_targets:
            return self._resolved_targets[s]
        _authorize_target_name(s, s, "hydra.logging")
        result = super().resolve(s)
        _reject_protected_result(result, s, "hydra.logging")
        self._authorize_callable(result, s)
        self._resolved_targets[s] = result
        self._resolved_target_sources[id(result)] = s
        return result

    def _prepare_custom_factory(self, config: Any) -> None:
        factory = config.get("()")
        if callable(factory):
            resolved_from = self._authorize_callable(factory, "")
        elif isinstance(factory, str):
            resolved_from = factory
            factory = self.resolve(factory)
        else:
            return

        def authorized_factory(*args: Any, **kwargs: Any) -> Any:
            return self._invoke_authorized_callable(
                factory, args, kwargs, resolved_from
            )

        config["()"] = authorized_factory

    def _apply_configured_properties(self, result: Any, props: Any) -> None:
        if props:
            for name, value in props.items():
                _authorize_target_invocation(
                    setattr,
                    (result, name, value),
                    {},
                    "hydra.logging",
                )
                setattr(result, name, value)

    def configure_custom(self, config: Any) -> Any:
        self._prepare_custom_factory(config)
        props = config.pop(".", None)
        result = super().configure_custom(config)
        self._apply_configured_properties(result, props)
        return result

    def _drop_invalid_formatter_result(self, result: Any, source: Any) -> Any:
        if not isinstance(result, logging.Formatter):
            warnings.warn(
                f"Logging formatter {source!r} returned "
                f"{type(result).__name__} instead of logging.Formatter; "
                "ignoring the configured formatter.",
                UserWarning,
                stacklevel=3,
            )
            return None
        return result

    def configure_formatter(self, config: Any) -> Any:
        if config.get("style", "%") == "{":
            raise InstantiationException(
                "Logging format style '{' cannot be selected by declarative "
                "configuration because its fields can traverse Python objects. "
                "Use '%' or '$' style, or configure logging from trusted Python code."
            )

        source = config.get("()", config.get("class", "logging.Formatter"))
        if "()" in config:
            result = super().configure_formatter(config)
        else:
            formatter_class = config.get("class")
            if isinstance(formatter_class, str):
                target = self.resolve(formatter_class)
                fmt = config.get("format")
                datefmt = config.get("datefmt")
                style = config.get("style", "%")
                args: Tuple[Any, ...] = (fmt, datefmt, style)
                if "validate" in config:
                    args += (config["validate"],)
                kwargs: Dict[str, Any] = {}
                if sys.version_info >= (3, 12):
                    defaults = config.get("defaults")
                    if defaults is not None:
                        kwargs["defaults"] = defaults
                result = self._invoke_authorized_callable(
                    target, args, kwargs, formatter_class
                )
            else:
                if callable(formatter_class):
                    self._authorize_callable(formatter_class, "")
                result = super().configure_formatter(config)
        return self._drop_invalid_formatter_result(result, source)

    def _configure_queue_handler(self, klass: Any, **kwargs: Any) -> Any:
        listener = kwargs.get("listener")
        if callable(listener):
            resolved_from = self._resolved_target_sources.get(id(listener))
            if resolved_from is None:
                resolved_from = self._authorize_callable(listener, "")

            def authorized_listener(*args: Any, **listener_kwargs: Any) -> Any:
                return self._invoke_authorized_callable(
                    listener, args, listener_kwargs, resolved_from
                )

            kwargs["listener"] = authorized_listener

        configure_queue_handler = getattr(super(), "_configure_queue_handler")
        return configure_queue_handler(klass, **kwargs)

    def configure_handler(self, config: Any) -> Any:
        if "()" in config:
            self._prepare_custom_factory(config)
        handler_class = config.get("class")
        resolved_from = ""
        queue_factory_path = ""
        if isinstance(handler_class, str):
            resolved_from = handler_class
            handler_class = self.resolve(handler_class)
        elif callable(handler_class):
            resolved_from = self._authorize_callable(handler_class, "")
        if callable(handler_class):
            kwargs = {
                key: value
                for key, value in config.items()
                if key not in {"class", "formatter", "level", "filters", "."}
                and key.isidentifier()
            }
            effective_target, effective_args, effective_kwargs = (
                _authorize_target_invocation(
                    handler_class,
                    (),
                    kwargs,
                    "hydra.logging",
                )
            )
            discovery_path = _authorize_discovery_path(
                effective_target,
                effective_args,
                effective_kwargs,
                "hydra.logging",
            )
            resolved_from = discovery_path or resolved_from
        for key in ("queue", "listener"):
            value = config.get(key)
            if callable(value):
                self._authorize_callable(value, "")

        deferred_config = {
            key: config.pop(key)
            for key in ("formatter", "level", "filters", ".")
            if key in config
        }
        try:
            if isinstance(handler_class, type) and issubclass(
                handler_class, logging.handlers.QueueHandler
            ):
                queue_factory = config.get("queue")
                if isinstance(queue_factory, str):
                    queue_factory_path = queue_factory
                    queue_target = self.resolve(queue_factory)
                    if not callable(queue_target):
                        raise TypeError(f"Invalid queue specifier {queue_factory!r}")
                    config["queue"] = self._invoke_authorized_callable(
                        queue_target, (), {}, queue_factory
                    )
            result = super().configure_handler(config)
        except Exception:
            if queue_factory_path:
                config["queue"] = queue_factory_path
            config.update(deferred_config)
            raise

        result = _mediate_target_result(
            result,
            resolved_from or _get_resolved_target_name_for_check(handler_class),
            "hydra.logging",
        )

        if not isinstance(result, logging.Handler):
            raise TypeError(
                "Configured handler factory must return a logging.Handler instance"
            )

        formatter = deferred_config.get("formatter")
        if formatter:
            try:
                formatter = self.config["formatters"][formatter]
            except Exception as exc:
                raise ValueError(f"Unable to set formatter {formatter!r}") from exc
            result.setFormatter(formatter)
        level = deferred_config.get("level")
        if level is not None:
            result.setLevel(level)
        filters = deferred_config.get("filters")
        if filters:
            self.add_filters(result, filters)
        props = deferred_config.get(".")
        self._apply_configured_properties(result, props)
        return result


def configure_logging(config: Dict[str, Any]) -> None:
    if logging.config.dictConfigClass is not logging.config.DictConfigurator:
        raise ValueError(
            dedent(
                """\
                Hydra does not support a custom logging.config.dictConfigClass
                because it can bypass Hydra target authorization. Express custom
                handlers, formatters, filters, queues, and listeners in the
                logging configuration instead."""
            )
        )
    HydraDictConfigurator(config).configure()
