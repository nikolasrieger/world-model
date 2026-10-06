import numpy as np
from PIL import Image, ImageDraw

BRICK_FIELDS = ((5, 6), (4, 6), (4, 4), (4, 2), (4, 0),
                (3, 0), (3, 2), (3, 4), (3, 6), (2, 4), (2, 6),
                (1, 6), (1, 4), (1, 2), (1, 0), (0, 0), (0, 2), (0, 4))
COLORS = ((200, 72, 72), (198, 108, 58), (180, 122, 48),
          (162, 162, 42), (72, 160, 72), (66, 72, 200))


def ram_image(state):
    values = np.asarray(state)
    if values.shape != (128,) or not np.isfinite(values).all():
        raise ValueError('Expected 128 finite normalized Breakout RAM values.')
    ram = np.rint(np.clip(values, 0, 1) * 255).astype(int)
    image = Image.new('RGB', (160, 210), 'black')
    draw = ImageDraw.Draw(image)
    for box in ((0, 18, 159, 31), (0, 32, 7, 193), (152, 32, 159, 193)):
        draw.rectangle(box, fill=(142, 142, 142))
    for row, color in enumerate(COLORS):
        for column, (group, bit) in enumerate(BRICK_FIELDS):
            if ram[group * 6 + 5 - row] & (1 << bit):
                x, y = 8 + 8 * column, 57 + 6 * row
                draw.rectangle((x, y, x + 7, y + 5), fill=color)
    x = ram[72] - 47
    draw.rectangle((x, 189, x + 15, 192), fill=COLORS[0])
    if 0 < ram[101] <= 187:
        x, y = ram[99] - 49, ram[101] + 9
        draw.rectangle((x, y, x + 1, y + 3), fill=COLORS[0])
    return image
