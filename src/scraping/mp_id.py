import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from mp_api.client import MPRester
import pandas as pd
from pymatgen.core import Structure
from src.config import DATA_PROCESSED

# 1. Setup your API Key
# Set the MATERIALS_PROJECT_API_KEY environment variable, or create a .env file
MAPI_KEY = os.environ.get("MATERIALS_PROJECT_API_KEY")
if not MAPI_KEY:
    raise ValueError(
        "Materials Project API key not found. "
        "Set the MATERIALS_PROJECT_API_KEY environment variable."
    )

print("Connecting to Materials Project...")

data_list = []

# 2. Query the Database
with MPRester(MAPI_KEY) as mpr:
    # Search for all materials containing Lithium ("Li")
    # specific queries can be added (e.g., is_stable=True for only stable materials)
    docs = mpr.materials.summary.search(
        elements=["Li"], 
        fields=["material_id", "formula_pretty", "structure", "density", "volume", "energy_above_hull"]
    )
    
    print(f"Found {len(docs)} Li-containing materials.")

    # 3. Extract relevant features for your model
    for doc in docs:
        stability = doc.energy_above_hull # 0 means stable

        if stability < 0.05:  # Filter for stable or near-stable materials
            # Extract basic info
            mp_id = doc.material_id
            formula = doc.formula_pretty
            
            # Structure object (Pymatgen structure)
            struct = doc.structure
            
            # Pre-calculated properties (useful for checking your descriptors)
            density = doc.density
            vol = doc.volume
            
            # Store in list
            data_list.append({
                "material_id": mp_id,
                "formula": formula,
                "density_g_cm3": density,
                "volume_A3": vol,
                "stability_eV_atom": stability,
                "structure_obj": struct,  # Keep full object if you need to calculate bond lengths/packing
                "cif": struct.to(fmt="cif") # Save CIF string if you need to export later
            })

# 4. Save to CSV (excluding the complex Structure object for simpler reading)
df = pd.DataFrame(data_list)

# Save a "clean" version for your descriptor generator
df_export = df.drop(columns=["structure_obj"]) 
df_export.to_csv(str(DATA_PROCESSED / "MP_Li_Materials.csv"), index=False)

print("Data saved to 'MP_Li_Materials.csv'.")
print("You can now loop through this CSV to generate your 14 physical descriptors + Mat2Vec embeddings.")