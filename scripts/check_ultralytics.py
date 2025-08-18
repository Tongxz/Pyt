try:
    import ultralytics

    print(f"ultralytics版本: {ultralytics.__version__}")
except ImportError:
    print("未安装ultralytics")
