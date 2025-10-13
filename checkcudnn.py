import pathlib, nvidia.cudnn as c
p = pathlib.Path(c.__file__).parent / "bin"
print(p)