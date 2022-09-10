# -*- coding: utf-8 -*-
"""
Created on Thu Sep  1 11:55:07 2022

@author: Leif Tinwell
"""
print("Starting")

import board

from kmk.kmk_keyboard import KMKKeyboard
from kmk.keys import KC
from kmk.scanners import DiodeOrientation

keyboard = KMKKeyboard()

keyboard.col_pins = (board.GP0,)    # try D5 on Feather, keeboar
keyboard.row_pins = (board.GP1,)    # try D6 on Feather, keeboar
keyboard.diode_orientation = DiodeOrientation.COL2ROW

keyboard.keymap = [
    [KC.A,]
]

if __name__ == '__main__':
    keyboard.go()
