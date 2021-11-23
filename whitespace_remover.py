import nbformat
from nbconvert.preprocessors import Preprocessor


class WhitespaceRemover(Preprocessor):
    """
    Remove trailing whitespace from code cells.
    """
    def preprocess_cell(self, cell: nbformat.NotebookNode, resources, index):
        if cell.cell_type == 'code' and cell.source:
            lines: list[str] = cell.source.split('\n')
            lines = [line.rstrip() for line in lines]
            cell.source = '\n'.join(lines)
        return cell, resources
