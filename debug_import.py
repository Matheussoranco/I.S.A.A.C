import sys
print(sys.path)
try:
    import mss
    print('mss import success')
except ImportError as e:
    print(f'mss import failed: {e}')
