# Outcome Reference Audit (12 October 2026)

This is an independent source-document audit by an AI assistant. It checks named penalty-event outcomes against FIFA's official report. It is not a new human video annotation, does not verify the identity of every video cut, and does not validate model accuracy or an unseen test set. No labels were modified.

The annotation table contains 39 distinct events. 39/39 binary references agree with official score progression. The stored table has 26 goals and 13 non-goals. The official PDF was retrieved from [FIFA Digital Hub](https://digitalhub.fifa.com/m/649e84967b086928/original/evdvpfdkueqrdlbbrrus-pdf.pdf), contains 140 pages and has SHA-256 `a5b871b2aef81b449608d62fb12274569e488c07aa3db83a812dfe4fabffa31b` (32,068,419 bytes). PDF page numbers and printed page numbers coincide.

The [stored legacy URL](https://img.fifa.com/image/upload/evdvpfdkueqrdlbbrrus.pdf) currently redirects to FIFA's document directory and returns HTML. That response is not the technical-report PDF. The independently retrieved Digital Hub file provides the checked primary-source evidence.

| Match | FIFA Match | Checked Page | Events | Goals | Non-goals |
| --- | ---: | ---: | ---: | ---: | ---: |
| col_eng | 56 | 134 | 10 | 7 | 3 |
| cro_den | 52 | 132 | 10 | 5 | 5 |
| esp_rus | 51 | 132 | 9 | 7 | 2 |
| rus_cro | 59 | 136 | 10 | 7 | 3 |

The audit parses each penalty table, starts the shootout at 0-0, and checks whether the named taker's team total increases by one. It also checks event order, country, shirt number and the clip-ID surname. Source accessibility, reference support and individual comparison are separate JSON fields. The comparison concerns the official event represented by an ID; it does not prove that all source cuts show the intended event.

The report's Denmark table prints `JORGENSEN Mathias (9)` for attempt 9. Its same-page substitution record instead identifies Nicolai Jorgensen as number 9. FIFA's [named-save account](https://inside.fifa.com/tournaments/mens/worldcup/2018russia/news/schmeichel-s-hat-trick-not-enough) also identifies Nicolai. The audit records this source-name inconsistency and checks Denmark, shirt 9 and attempt order without changing the clip ID or outcome. `KUZIAEV` and `kuzyaev` are recorded as a transliteration difference.

A binary non-goal does not by itself establish `saved`, `crossbar` or `wide`. The PDF explicitly describes Henderson's save. FIFA's named-save account supports the five Denmark-Croatia saves; its [Spain-Russia account](https://inside.fifa.com/tournaments/mens/worldcup/2018russia/news/russia-win-first-ever-world-cup-shootout-to-reach-quarters-2980138) names the saves of Koke and Aspas. The checked sources do not individually establish the stored detailed mechanism for the following five events: Uribe (`crossbar`), Bacca (`saved`), Russia-Croatia Smolov (`saved`), Kovacic (`saved`) and Fernandes (`wide`). Their binary outcomes are supported; their detailed mechanisms remain unknown in this audit.

| Event | Page | Official Score Before To After | Stored Outcome | Accessible | Supported | Individually Checked |
| --- | ---: | --- | --- | --- | --- | --- |
| col_eng_p01_falcao_col9 | 134 | 0-0 to 1-0 | goal | yes | yes | yes |
| col_eng_p02_kane_eng9 | 134 | 1-0 to 1-1 | goal | yes | yes | yes |
| col_eng_p03_cuadrado_col11 | 134 | 1-1 to 2-1 | goal | yes | yes | yes |
| col_eng_p04_rashford_eng19 | 134 | 2-1 to 2-2 | goal | yes | yes | yes |
| col_eng_p05_muriel_col14 | 134 | 2-2 to 3-2 | goal | yes | yes | yes |
| col_eng_p06_henderson_eng8 | 134 | 3-2 to 3-2 | no_goal | yes | yes | yes |
| col_eng_p07_uribe_col15 | 134 | 3-2 to 3-2 | no_goal | yes | yes | yes |
| col_eng_p08_trippier_eng12 | 134 | 3-2 to 3-3 | goal | yes | yes | yes |
| col_eng_p09_bacca_col7 | 134 | 3-3 to 3-3 | no_goal | yes | yes | yes |
| col_eng_p10_dier_eng4 | 134 | 3-3 to 3-4 | goal | yes | yes | yes |
| cro_den_p01_eriksen_den10 | 132 | 0-0 to 0-0 | no_goal | yes | yes | yes |
| cro_den_p02_badelj_cro19 | 132 | 0-0 to 0-0 | no_goal | yes | yes | yes |
| cro_den_p03_kjaer_den4 | 132 | 0-0 to 0-1 | goal | yes | yes | yes |
| cro_den_p04_kramaric_cro9 | 132 | 0-1 to 1-1 | goal | yes | yes | yes |
| cro_den_p05_krohndehli_den2 | 132 | 1-1 to 1-2 | goal | yes | yes | yes |
| cro_den_p06_modric_cro10 | 132 | 1-2 to 2-2 | goal | yes | yes | yes |
| cro_den_p07_schone_den19 | 132 | 2-2 to 2-2 | no_goal | yes | yes | yes |
| cro_den_p08_pivaric_cro22 | 132 | 2-2 to 2-2 | no_goal | yes | yes | yes |
| cro_den_p09_jorgensen_den9 | 132 | 2-2 to 2-2 | no_goal | yes | yes | yes |
| cro_den_p10_rakitic_cro7 | 132 | 2-2 to 3-2 | goal | yes | yes | yes |
| esp_rus_p01_iniesta_esp6 | 132 | 0-0 to 1-0 | goal | yes | yes | yes |
| esp_rus_p02_smolov_rus10 | 132 | 1-0 to 1-1 | goal | yes | yes | yes |
| esp_rus_p03_pique_esp3 | 132 | 1-1 to 2-1 | goal | yes | yes | yes |
| esp_rus_p04_ignashevich_rus4 | 132 | 2-1 to 2-2 | goal | yes | yes | yes |
| esp_rus_p05_koke_esp8 | 132 | 2-2 to 2-2 | no_goal | yes | yes | yes |
| esp_rus_p06_golovin_rus17 | 132 | 2-2 to 2-3 | goal | yes | yes | yes |
| esp_rus_p07_ramos_esp15 | 132 | 2-3 to 3-3 | goal | yes | yes | yes |
| esp_rus_p08_cheryshev_rus6 | 132 | 3-3 to 3-4 | goal | yes | yes | yes |
| esp_rus_p09_aspas_esp17 | 132 | 3-4 to 3-4 | no_goal | yes | yes | yes |
| rus_cro_p01_smolov_rus10 | 136 | 0-0 to 0-0 | no_goal | yes | yes | yes |
| rus_cro_p02_brozovic_cro11 | 136 | 0-0 to 0-1 | goal | yes | yes | yes |
| rus_cro_p03_dzagoev_rus9 | 136 | 0-1 to 1-1 | goal | yes | yes | yes |
| rus_cro_p04_kovacic_cro8 | 136 | 1-1 to 1-1 | no_goal | yes | yes | yes |
| rus_cro_p05_fernandes_rus2 | 136 | 1-1 to 1-1 | no_goal | yes | yes | yes |
| rus_cro_p06_modric_cro10 | 136 | 1-1 to 1-2 | goal | yes | yes | yes |
| rus_cro_p07_ignashevich_rus4 | 136 | 1-2 to 2-2 | goal | yes | yes | yes |
| rus_cro_p08_vida_cro21 | 136 | 2-2 to 2-3 | goal | yes | yes | yes |
| rus_cro_p09_kuzyaev_rus7 | 136 | 2-3 to 3-3 | goal | yes | yes | yes |
| rus_cro_p10_rakitic_cro7 | 136 | 3-3 to 3-4 | goal | yes | yes | yes |

The authoritative machine-readable record is [OUTCOME_REFERENCE_AUDIT_2026-10-12.json](OUTCOME_REFERENCE_AUDIT_2026-10-12.json). The retrieved PDF and three rendered evidence pages are retained in `outputs/reference_source_audit_2026-10-12/`. Reproduce the binary comparison with `scripts/audit_outcome_references.py --official-pdf <retrieved-official-PDF>`; supplementary article findings were independently read during this audit and remain explicitly identified as source-reading records.
