import os
import sys

# Prevent OpenMP DLL conflict between PyTorch and other libraries
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["KMP_BLOCKTIME"] = "0"

# Tell PyTorch where its own DLLs live inside the frozen bundle
if getattr(sys, 'frozen', False):
    # _MEIPASS is the _internal folder PyInstaller extracts/uses
    base_dir = sys._MEIPASS
    torch_lib = os.path.join(base_dir, "torch", "lib")
    
    if os.path.isdir(torch_lib):
        # Python 3.8+ ignores PATH for DLL resolution. We MUST use add_dll_directory
        # to ensure c10.dll can find msvcp140.dll (in base_dir) and itself.
        if hasattr(os, 'add_dll_directory'):
            os.add_dll_directory(base_dir)
            os.add_dll_directory(torch_lib)
            
        # Also keeping PATH fallback for legacy/external tools
        os.environ["PATH"] = base_dir + os.pathsep + torch_lib + os.pathsep + os.environ.get("PATH", "")
