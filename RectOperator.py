class Rect:
    def __init__(self, left, top, right, bottom):
        self.left = left
        self.top = top
        self.right = right
        self.bottom = bottom

    def expand(self, x, y):
        self.left = min([self.left, x])
        self.top = min([self.top, y])
        self.right = max([self.right, x])
        self.bottom = max([self.bottom, y])

    def get_width(self):
        return self.right - self.left + 1

    def get_height(self):
        return self.bottom - self.top + 1

    def get_area(self):
        return self.get_width() * self.get_height()

def is_intersect(rt1, rt2):
    if rt1.right < rt2.left or rt1.left > rt2.right:
        return False
    if rt1.bottom < rt2.top or rt1.top > rt2.bottom:
        return False
    return True

def is_same(rt1, rt2):
    if rt1.left == rt2.left and rt1.top == rt2.top and rt1.right == rt2.right and rt1.bottom == rt2.bottom:
        return True
    return False