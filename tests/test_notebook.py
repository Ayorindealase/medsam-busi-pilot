import ast
from pathlib import Path
import nbformat


def test_notebook_schema_and_python_cells():
    notebook = nbformat.read(Path(__file__).parents[1] / 'notebooks/BUSI_MedSAM_Pilot.ipynb', as_version=4)
    nbformat.validate(notebook)
    for cell in notebook.cells:
        if cell.cell_type == 'code':
            ast.parse(cell.source)
            assert cell.execution_count is None
            assert not cell.outputs
