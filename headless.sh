python -m user_pipeline.batch_cli \
  --checkpoint weights/noklamath.pt \
  --output-dir /mnt/data/tardis_cpu_outputs \
  --split test \
  --split-path /mnt/data/CFC26_MAH/splits.json \
  --aris-super-dir /mnt/data/CFC26_MAH/aris \
#   --no-csv \
#   --locations kenai-channel-stratum1 \
#   --no-filter-submasks \
#   --imgsz 1280 \
#   --conf 0.99 \
#   --no-png \
