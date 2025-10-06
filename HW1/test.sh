RUN="y11m_stage1_freeze10"
W="runs/pig/${RUN}_tight/weights/best.pt"; [ -f "$W" ] || W="runs/pig/${RUN}_tight/weights/last.pt"
python inference.py --weights "$W" --test_dir data/test/images --imgsz 640 --conf 0.2 --tta --out_csv submission.csv