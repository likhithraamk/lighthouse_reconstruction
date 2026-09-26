from PIL import Image
from PIL.ExifTags import TAGS
import glob, os

paths = sorted(glob.glob("images/*.jpg") + glob.glob("images/*.JPG"))
seen, unique = set(), []
for p in paths:
    n = os.path.basename(p)
    if n not in seen:
        seen.add(n); unique.append(p)

for p in unique[:3]:
    img = Image.open(p)
    exif = img._getexif()
    print(f"\n{os.path.basename(p)} — size: {img.size}")
    if exif:
        for tag_id, val in exif.items():
            tag = TAGS.get(tag_id, tag_id)
            if tag in ('FocalLength','FocalLengthIn35mmFilm','Make','Model',
                       'ExifImageWidth','ExifImageHeight','PixelXDimension','PixelYDimension'):
                print(f"  {tag}: {val}")
    else:
        print("  No EXIF data")