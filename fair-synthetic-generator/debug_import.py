try:
    import src.training
    print("Import successful")
except SyntaxError as e:
    print(f"SyntaxError: {e}")
    print(f"File: {e.filename}")
    print(f"Line: {e.lineno}")
    print(f"Offset: {e.offset}")
    print(f"Text: {e.text}")
except Exception as e:
    print(f"Exception: {e}")
    import traceback
    traceback.print_exc()
