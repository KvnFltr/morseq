"""Check every statement of the report on random fields (any grid shape)."""
from morseq.cli import main

main(["check", "--shape", "5x4", "--trials", "30"])
main(["check", "--shape", "3x3x3", "--trials", "10"])
