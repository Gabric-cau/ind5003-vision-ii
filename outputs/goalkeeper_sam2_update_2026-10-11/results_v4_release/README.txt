SAM2门将v4接入包。先打开 report.html。
这是方向候选更新，不是人工真值/模型准确率；height/whole-record QA不因方向恢复而提升。
新增主入口：scripts/run_goalkeeper_sam2_update.py；--head-part-fallback 开启自动头部子区域续跟。
默认SAM2分辨率1024；六例均试过512，但Cuadrado保留1024身体+512头部，其他5例为512身体。CPU推理，CUDA扩展关闭。
在原项目根目录运行，必须选择新的输出目录：
python -X utf8 scripts/run_goalkeeper_sam2_update.py --image-size 512 --head-part-fallback --output-dir outputs/my_sam2_run
--clips 可只测指定案例；不传时只处理旧v3的6条unknown。已知33条不重新分类。
--resume 只允许协议/源码一致且输入哈希未变的已有运行，不应绕过校验。
本次Cuadrado执行头部fallback；另5条实际仅运行身体+姿态。不要把此发布声称为所有6条头部fallback都已跑过。
原v3保留在 outputs/goalkeeper_pose_direction_2026-10-11/results_v3_final。
上游缓存、原视频和本地权重为依赖，包中不附150MB模型或完整原视频；原视频 data/raw_clips。
Meta权重：https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt
官方SAM2提交：2b90b9f5ceec907a1c18123530e92e794ad901a4；隔离依赖 .deps_goalkeeper_segmentation。
所有新方法源码在source_snapshot，原身体批次源码在body_batch_source_snapshot（与当时协议哈希一致）。
源片段截尾/ROI边缘/裁判遮挡/落地滚动都可能导致unknown或错误方向，仍需复核。
