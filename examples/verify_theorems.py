"""Check the statements of the README on random Morse sequences, in 2-D and 3-D."""
from morseq.cli import main

main(["check", "--shape", "5x4", "--trials", "30"])
main(["check", "--shape", "3x3x3", "--trials", "10"])
