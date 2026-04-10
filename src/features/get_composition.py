import pandas as pd
import re
import numpy as np
import math
from collections import defaultdict

# Extended periodic table for battery materials
PERIODIC_TABLE = {
    'H': {'atomic_mass': 1.008, 'electronegativity': 2.20, 'ionic_radius': 0.37, 'group': 1},
    'Li': {'atomic_mass': 6.94, 'electronegativity': 0.98, 'ionic_radius': 0.76, 'group': 1},
    'Be': {'atomic_mass': 9.01, 'electronegativity': 1.57, 'ionic_radius': 0.45, 'group': 2},
    'B': {'atomic_mass': 10.81, 'electronegativity': 2.04, 'ionic_radius': 0.27, 'group': 13},
    'C': {'atomic_mass': 12.01, 'electronegativity': 2.55, 'ionic_radius': 0.16, 'group': 14},
    'N': {'atomic_mass': 14.01, 'electronegativity': 3.04, 'ionic_radius': 0.13, 'group': 15},
    'O': {'atomic_mass': 16.00, 'electronegativity': 3.44, 'ionic_radius': 1.40, 'group': 16},
    'F': {'atomic_mass': 19.00, 'electronegativity': 3.98, 'ionic_radius': 1.33, 'group': 17},
    'Na': {'atomic_mass': 22.99, 'electronegativity': 0.93, 'ionic_radius': 1.02, 'group': 1},
    'Mg': {'atomic_mass': 24.31, 'electronegativity': 1.31, 'ionic_radius': 0.72, 'group': 2},
    'Al': {'atomic_mass': 26.98, 'electronegativity': 1.61, 'ionic_radius': 0.54, 'group': 13},
    'Si': {'atomic_mass': 28.09, 'electronegativity': 1.90, 'ionic_radius': 0.40, 'group': 14},
    'P': {'atomic_mass': 30.97, 'electronegativity': 2.19, 'ionic_radius': 0.44, 'group': 15},
    'S': {'atomic_mass': 32.07, 'electronegativity': 2.58, 'ionic_radius': 1.84, 'group': 16},
    'Cl': {'atomic_mass': 35.45, 'electronegativity': 3.16, 'ionic_radius': 1.81, 'group': 17},
    'K': {'atomic_mass': 39.10, 'electronegativity': 0.82, 'ionic_radius': 1.38, 'group': 1},
    'Ca': {'atomic_mass': 40.08, 'electronegativity': 1.00, 'ionic_radius': 1.00, 'group': 2},
    'Ti': {'atomic_mass': 47.87, 'electronegativity': 1.54, 'ionic_radius': 0.61, 'group': 4},
    'V': {'atomic_mass': 50.94, 'electronegativity': 1.63, 'ionic_radius': 0.58, 'group': 5},
    'Cr': {'atomic_mass': 52.00, 'electronegativity': 1.66, 'ionic_radius': 0.52, 'group': 6},
    'Mn': {'atomic_mass': 54.94, 'electronegativity': 1.55, 'ionic_radius': 0.67, 'group': 7},
    'Fe': {'atomic_mass': 55.85, 'electronegativity': 1.83, 'ionic_radius': 0.65, 'group': 8},
    'Co': {'atomic_mass': 58.93, 'electronegativity': 1.88, 'ionic_radius': 0.65, 'group': 9},
    'Ni': {'atomic_mass': 58.69, 'electronegativity': 1.91, 'ionic_radius': 0.69, 'group': 10},
    'Cu': {'atomic_mass': 63.55, 'electronegativity': 1.90, 'ionic_radius': 0.73, 'group': 11},
    'Zn': {'atomic_mass': 65.38, 'electronegativity': 1.65, 'ionic_radius': 0.74, 'group': 12},
    'Ga': {'atomic_mass': 69.72, 'electronegativity': 1.81, 'ionic_radius': 0.62, 'group': 13},
    'Ge': {'atomic_mass': 72.63, 'electronegativity': 2.01, 'ionic_radius': 0.53, 'group': 14},
    'As': {'atomic_mass': 74.92, 'electronegativity': 2.18, 'ionic_radius': 0.58, 'group': 15},
    'Se': {'atomic_mass': 78.97, 'electronegativity': 2.55, 'ionic_radius': 1.98, 'group': 16},
    'Br': {'atomic_mass': 79.90, 'electronegativity': 2.96, 'ionic_radius': 1.96, 'group': 17},
    'Rb': {'atomic_mass': 85.47, 'electronegativity': 0.82, 'ionic_radius': 1.52, 'group': 1},
    'Sr': {'atomic_mass': 87.62, 'electronegativity': 0.95, 'ionic_radius': 1.18, 'group': 2},
    'Y': {'atomic_mass': 88.91, 'electronegativity': 1.22, 'ionic_radius': 0.90, 'group': 3},
    'Zr': {'atomic_mass': 91.22, 'electronegativity': 1.33, 'ionic_radius': 0.72, 'group': 4},
    'Nb': {'atomic_mass': 92.91, 'electronegativity': 1.6, 'ionic_radius': 0.64, 'group': 5},
    'Mo': {'atomic_mass': 95.96, 'electronegativity': 2.16, 'ionic_radius': 0.65, 'group': 6},
    'Ag': {'atomic_mass': 107.87, 'electronegativity': 1.93, 'ionic_radius': 1.15, 'group': 11},
    'Cd': {'atomic_mass': 112.41, 'electronegativity': 1.69, 'ionic_radius': 0.95, 'group': 12},
    'In': {'atomic_mass': 114.82, 'electronegativity': 1.78, 'ionic_radius': 0.80, 'group': 13},
    'Sn': {'atomic_mass': 118.71, 'electronegativity': 1.96, 'ionic_radius': 0.69, 'group': 14},
    'Sb': {'atomic_mass': 121.76, 'electronegativity': 2.05, 'ionic_radius': 0.76, 'group': 15},
    'Te': {'atomic_mass': 127.60, 'electronegativity': 2.1, 'ionic_radius': 2.21, 'group': 16},
    'I': {'atomic_mass': 126.90, 'electronegativity': 2.66, 'ionic_radius': 2.20, 'group': 17},
    'Cs': {'atomic_mass': 132.91, 'electronegativity': 0.79, 'ionic_radius': 1.67, 'group': 1},
    'Ba': {'atomic_mass': 137.33, 'electronegativity': 0.89, 'ionic_radius': 1.35, 'group': 2},
    'La': {'atomic_mass': 138.91, 'electronegativity': 1.10, 'ionic_radius': 1.03, 'group': 3},
    'Ce': {'atomic_mass': 140.12, 'electronegativity': 1.12, 'ionic_radius': 1.01, 'group': 3},
    'Nd': {'atomic_mass': 144.24, 'electronegativity': 1.14, 'ionic_radius': 0.98, 'group': 3},
    'Sm': {'atomic_mass': 150.36, 'electronegativity': 1.17, 'ionic_radius': 0.96, 'group': 3},
    'Eu': {'atomic_mass': 151.96, 'electronegativity': 1.2, 'ionic_radius': 0.95, 'group': 3},
    'Gd': {'atomic_mass': 157.25, 'electronegativity': 1.20, 'ionic_radius': 0.94, 'group': 3},
    'Tb': {'atomic_mass': 158.93, 'electronegativity': 1.2, 'ionic_radius': 0.92, 'group': 3},
    'Dy': {'atomic_mass': 162.50, 'electronegativity': 1.22, 'ionic_radius': 0.91, 'group': 3},
    'Ho': {'atomic_mass': 164.93, 'electronegativity': 1.23, 'ionic_radius': 0.90, 'group': 3},
    'Er': {'atomic_mass': 167.26, 'electronegativity': 1.24, 'ionic_radius': 0.89, 'group': 3},
    'Tm': {'atomic_mass': 168.93, 'electronegativity': 1.25, 'ionic_radius': 0.88, 'group': 3},
    'Yb': {'atomic_mass': 173.05, 'electronegativity': 1.1, 'ionic_radius': 0.87, 'group': 3},
    'Lu': {'atomic_mass': 174.97, 'electronegativity': 1.27, 'ionic_radius': 0.86, 'group': 3},
    'Ta': {'atomic_mass': 180.95, 'electronegativity': 1.5, 'ionic_radius': 0.64, 'group': 5},
    'W': {'atomic_mass': 183.84, 'electronegativity': 2.36, 'ionic_radius': 0.66, 'group': 6},
    'Pb': {'atomic_mass': 207.2, 'electronegativity': 2.33, 'ionic_radius': 1.19, 'group': 14},
    'Bi': {'atomic_mass': 208.98, 'electronegativity': 2.02, 'ionic_radius': 1.03, 'group': 15},
    'Hf': {'atomic_mass': 178.49, 'electronegativity': 1.3, 'ionic_radius': 0.70, 'group': 4},
    'Sc': {'atomic_mass': 44.96, 'electronegativity': 1.36, 'ionic_radius': 0.81, 'group': 3},
    'Pr': {'atomic_mass': 140.91, 'electronegativity': 1.13, 'ionic_radius': 0.99, 'group': 3},
    'Rh': {'atomic_mass': 102.91, 'electronegativity': 2.28, 'ionic_radius': 0.67, 'group': 9},
    'U': {'atomic_mass': 238.03, 'electronegativity': 1.38, 'ionic_radius': 1.00, 'group': None},
    'Ru': {'atomic_mass': 101.07, 'electronegativity': 2.2, 'ionic_radius': 0.68, 'group': 8},
    'Tl': {'atomic_mass': 204.38, 'electronegativity': 1.62, 'ionic_radius': 0.88, 'group': 13},
    "Hg": {'atomic_mass': 200.59, 'electronegativity': 2.00, 'ionic_radius': 1.02, 'group': 12},
    'Pd': {'atomic_mass': 106.42, 'electronegativity': 2.20, 'ionic_radius': 0.69, 'group': 10},
    'Pt': {'atomic_mass': 195.08, 'electronegativity': 2.28, 'ionic_radius': 0.72, 'group': 10},
    'Au': {'atomic_mass': 196.97, 'electronegativity': 2.54, 'ionic_radius': 1.00, 'group': 11},
    'Re': {'atomic_mass': 186.21, 'electronegativity': 1.9, 'ionic_radius': 0.63, 'group': 7},
    'Np': {'atomic_mass': 237.05, 'electronegativity': 1.36, 'ionic_radius': 1.01, 'group': None},
    'Os': {'atomic_mass': 190.23, 'electronegativity': 2.2, 'ionic_radius': 0.69, 'group': 8},
    'Ir': {'atomic_mass': 192.22, 'electronegativity': 2.20, 'ionic_radius': 0.72, 'group': 9},
    'Xe': {'atomic_mass': 131.29, 'electronegativity': 2.60, 'ionic_radius': 2.16, 'group': 18},
    'Pm': {'atomic_mass': 145.0, 'electronegativity': 1.13, 'ionic_radius': 0.97, 'group': 3},
    'Ac': {'atomic_mass': 227.03, 'electronegativity': 1.1, 'ionic_radius': 1.12, 'group': None},
    'Tc': {'atomic_mass': 98.0, 'electronegativity': 1.9, 'ionic_radius': 0.64, 'group': 7},
    'Th': {'atomic_mass': 232.04, 'electronegativity': 1.3, 'ionic_radius': 1.19, 'group': None},
    'Pa': {'atomic_mass': 231.04, 'electronegativity': 1.5, 'ionic_radius': 1.14, 'group': None},
}

def preprocess_formula(formula):
    """
    Clean formula by removing descriptors in brackets at the end
    """
    if not isinstance(formula, str):
        return ""
    
    formula = formula.strip()
    
    # Remove descriptors in parentheses/brackets at the end (not chemical formulas)
    # Keep chemical parentheses like (BH4)2 but remove (ball milled), [dried], etc.
    formula = re.sub(r'\s*[\(\[\{][^\)\]\}]*?(?:milled|dried|heated|ground|annealed|treated|prepared|processed)[^\)\]\}]*?[\)\]\}]\s*$', '', formula, flags=re.IGNORECASE)
    
    return formula.strip()

def parse_complex_formula(formula):
    """
    Enhanced recursive parser for complex nested formulas with decimal multipliers
    Handles: ((Li2S)0.75(P2S5)0.25)96(P2O5)4
    """
    formula = formula.replace(' ', '')
    
    def _parse_segment(segment, multiplier=1.0):
        composition = defaultdict(float)
        i = 0
        
        while i < len(segment):
            if segment[i] == '(':
                # Find matching closing parenthesis
                level = 1
                j = i + 1
                while j < len(segment) and level > 0:
                    if segment[j] == '(':
                        level += 1
                    elif segment[j] == ')':
                        level -= 1
                    j += 1
                
                if level != 0:
                    raise ValueError(f"Unmatched parenthesis in formula: {segment}")
                
                # Extract content inside parentheses
                subgroup = segment[i+1:j-1]
                
                # Look for multiplier after closing parenthesis
                k = j
                multiplier_str = ''
                while k < len(segment) and (segment[k].isdigit() or segment[k] == '.'):
                    multiplier_str += segment[k]
                    k += 1
                
                sub_multiplier = float(multiplier_str) if multiplier_str else 1.0
                
                # Recursively parse subgroup
                sub_comp = _parse_segment(subgroup, multiplier * sub_multiplier)
                
                # Add to main composition
                for elem, count in sub_comp.items():
                    composition[elem] += count
                
                i = k
                
            else:
                # Parse element and its count
                elem_match = re.match(r'([A-Z][a-z]?)([0-9]*\.?[0-9]*)', segment[i:])
                if not elem_match:
                    # Skip non-element characters or break
                    i += 1
                    continue
                
                element = elem_match.group(1)
                count_str = elem_match.group(2)
                count = float(count_str) if count_str else 1.0
                
                composition[element] += count * multiplier
                i += len(elem_match.group(0))
        
        return composition
    
    try:
        result = _parse_segment(formula)
        return dict(result)
    except Exception as e:
        raise ValueError(f"Failed to parse formula '{formula}': {e}")

def parse_mixture_formula(formula):
    """
    Parse mixture formulas like "0.75(LiBH4)-0.25Ca(BH4)2"
    """
    formula = preprocess_formula(formula)
    
    if not formula:
        return {}
    
    # Check if it's a mixture (contains separators with numbers)
    mixture_pattern = r'[0-9]\s*[-+·•:]'
    is_mixture = bool(re.search(mixture_pattern, formula))
    
    if not is_mixture:
        # Single compound, parse directly
        return parse_complex_formula(formula)
    
    # Split on mixture separators while preserving the fractions
    mixture_separators = ['-', '+', '·', '•', ':']
    
    # Find all separator positions
    separator_positions = []
    for i, char in enumerate(formula):
        if char in mixture_separators:
            # Check if there's a number before this separator
            j = i - 1
            while j >= 0 and formula[j].isspace():
                j -= 1
            if j >= 0 and (formula[j].isdigit() or formula[j] == ')'):
                separator_positions.append(i)
    
    # Split the formula at separator positions
    if not separator_positions:
        # No valid separators found, treat as single compound
        return parse_complex_formula(formula)
    
    parts = []
    start = 0
    for pos in separator_positions:
        parts.append(formula[start:pos].strip())
        start = pos + 1
    parts.append(formula[start:].strip())  # Last part
    
    # Parse each part
    total_composition = defaultdict(float)
    
    for part in parts:
        if not part:
            continue
            
        # Extract fraction prefix if present
        fraction_match = re.match(r'^([0-9]*\.?[0-9]+)\s*(.*)$', part)
        if fraction_match:
            fraction = float(fraction_match.group(1))
            chemical_part = fraction_match.group(2)
        else:
            fraction = 1.0
            chemical_part = part
        
        # Parse the chemical formula
        part_composition = parse_complex_formula(chemical_part)
        
        # Add to total with fraction weighting
        for element, count in part_composition.items():
            total_composition[element] += count * fraction
    
    return dict(total_composition)

def calculate_weighted_property(composition, property_name):
    """Calculate weighted average of elemental property"""
    total_atoms = sum(composition.values())
    if total_atoms == 0:
        return None
    
    weighted_sum = 0
    missing_elements = []
    
    for element, count in composition.items():
        if element in PERIODIC_TABLE and property_name in PERIODIC_TABLE[element]:
            weighted_sum += PERIODIC_TABLE[element][property_name] * count
        else:
            missing_elements.append(element)
    
    if missing_elements:
        return None
    
    return weighted_sum / total_atoms

def calculate_composition_entropy(composition):
    """Calculate configurational entropy"""
    total_atoms = sum(composition.values())
    if total_atoms == 0:
        return 0
    
    entropy = 0
    for count in composition.values():
        fraction = count / total_atoms
        if fraction > 0:
            entropy -= fraction * np.log(fraction)
    
    return entropy

def calculate_electronegativity_variance(composition):
    """Calculate variance in electronegativity"""
    electronegativities = []
    for element, count in composition.items():
        if element in PERIODIC_TABLE and 'electronegativity' in PERIODIC_TABLE[element]:
            # Weight by count (use ceiling to ensure at least 1)
            weight = max(1, int(math.ceil(count)))
            electronegativities.extend([PERIODIC_TABLE[element]['electronegativity']] * weight)
    
    return np.var(electronegativities) if len(electronegativities) > 1 else 0

def get_element_group_diversity(composition):
    """Calculate number of unique periodic groups"""
    groups = set()
    for element in composition.keys():
        if element in PERIODIC_TABLE and 'group' in PERIODIC_TABLE[element]:
            groups.add(PERIODIC_TABLE[element]['group'])
    return len(groups)

def calculate_packing_efficiency_proxy(composition):
    """Estimate packing efficiency using ionic radii"""
    radii = []
    for element, count in composition.items():
        if element in PERIODIC_TABLE and 'ionic_radius' in PERIODIC_TABLE[element]:
            weight = max(1, int(math.ceil(count)))
            radii.extend([PERIODIC_TABLE[element]['ionic_radius']] * weight)
    
    if not radii:
        return None
    
    return min(radii) / max(radii) if max(radii) > 0 else None

def enhance_composition_features_fixed(df, formula_column='electrolyte'):
    """
    FIXED: Enhanced function to add chemical composition features
    Properly handles complex nested formulas and mixtures
    """
    print(f"Processing {len(df)} compounds with advanced formula parsing...")
    print(f"Formula column: '{formula_column}'")
    
    # Validation
    if formula_column not in df.columns:
        raise ValueError(f"Column '{formula_column}' not found. Available: {list(df.columns)}")
    
    # Initialize feature lists
    features = {
        'avg_electronegativity': [],
        'avg_atomic_mass': [],
        'avg_ionic_radius': [],
        'num_elements': [],
        'li_fraction': [],
        'composition_entropy': [],
        'electronegativity_variance': [],
        'group_diversity': [],
        'packing_efficiency_proxy': [],
        'li_to_anion_ratio': [],
        'heaviest_element_mass': [],
        'lightest_element_mass': [],
        'is_mixture': [],
        'formula_complexity': [],
        'total_atoms': []
    }
    
    # Statistics tracking
    processed_count = 0
    error_count = 0
    mixture_count = 0
    empty_count = 0
    missing_elements_count = 0
    complex_formula_count = 0
    
    print("\nProcessing formulas...")
    print("-" * 60)
    
    for i, formula in enumerate(df[formula_column]):
        try:
            # Progress tracking
            if (i + 1) % max(1, len(df) // 20) == 0:
                progress = (i + 1) / len(df) * 100
                print(f"Progress: {progress:.1f}% ({i + 1}/{len(df)}) - "
                      f"Errors: {error_count}, Mixtures: {mixture_count}, Complex: {complex_formula_count}")
            
            # Handle empty/invalid formulas
            if pd.isna(formula) or not isinstance(formula, str) or not formula.strip():
                print(f"Warning: Empty formula at row {i}")
                empty_count += 1
                for feature_name in features.keys():
                    features[feature_name].append(None)
                continue
            
            # Check formula characteristics
            is_mixture = any(sep in formula for sep in ['-', '+', '·', '•', ':', '-']) and re.search(r'[0-9]\s*[-+·•:]', formula)
            is_complex = '((' in formula or ')(' in formula or re.search(r'\)[0-9]*\.?[0-9]*\(', formula)
            
            if is_mixture:
                mixture_count += 1
            if is_complex:
                complex_formula_count += 1
            
            # Parse composition using the fixed parser
            composition = parse_mixture_formula(formula)
            
            if not composition:
                print(f"Warning: No elements detected in '{formula}' at row {i}")
                empty_count += 1
                for feature_name in features.keys():
                    features[feature_name].append(None)
                continue
            
            # Check for missing elements in periodic table
            missing_elements = [elem for elem in composition.keys() if elem not in PERIODIC_TABLE]
            if missing_elements:
                print(f"Warning: Unknown elements {missing_elements} in '{formula}' at row {i}")
                missing_elements_count += 1
            
            # Calculate features
            total_atoms = sum(composition.values())
            
            # Basic weighted averages
            features['avg_electronegativity'].append(
                calculate_weighted_property(composition, 'electronegativity')
            )
            features['avg_atomic_mass'].append(
                calculate_weighted_property(composition, 'atomic_mass')
            )
            features['avg_ionic_radius'].append(
                calculate_weighted_property(composition, 'ionic_radius')
            )
            
            # Composition metrics
            features['num_elements'].append(len(composition))
            features['li_fraction'].append(
                composition.get('Li', 0) / total_atoms if total_atoms > 0 else 0
            )
            features['composition_entropy'].append(
                calculate_composition_entropy(composition)
            )
            features['electronegativity_variance'].append(
                calculate_electronegativity_variance(composition)
            )
            features['group_diversity'].append(
                get_element_group_diversity(composition)
            )
            features['packing_efficiency_proxy'].append(
                calculate_packing_efficiency_proxy(composition)
            )
            
            # Li-specific ratios
            li_count = composition.get('Li', 0)
            anion_count = sum(composition.get(elem, 0) for elem in ['O', 'S', 'Cl', 'F', 'Br', 'I'])
            features['li_to_anion_ratio'].append(
                li_count / anion_count if anion_count > 0 else 0
            )
            
            # Mass extremes
            masses = [PERIODIC_TABLE[elem]['atomic_mass'] for elem in composition.keys() 
                     if elem in PERIODIC_TABLE]
            features['heaviest_element_mass'].append(max(masses) if masses else None)
            features['lightest_element_mass'].append(min(masses) if masses else None)
            
            # Additional features
            features['is_mixture'].append(is_mixture)
            # change is_mixture to 0 or 1
            features['is_mixture'][-1] = 0 if not is_mixture else 1
            features['formula_complexity'].append(len(composition) * total_atoms)
            features['total_atoms'].append(total_atoms)
            
            processed_count += 1
            
            # Debug output for complex formulas
            if is_complex and i < 5:  # Show first 5 complex formulas
                print(f"Complex formula example: '{formula}' -> Li fraction: {composition.get('Li', 0) / total_atoms:.3f}")
            
        except Exception as e:
            print(f"Error processing '{formula}' at row {i}: {e}")
            error_count += 1
            for feature_name in features.keys():
                features[feature_name].append(None)
    
    # Add all features to dataframe
    for feature_name, feature_values in features.items():
        df[feature_name] = feature_values
    
    # Print comprehensive summary
    print("\n" + "="*70)
    print("PROCESSING SUMMARY")
    print("="*70)
    print(f"Total formulas: {len(df)}")
    print(f"Successfully processed: {processed_count}")
    print(f"Errors: {error_count}")
    print(f"Empty/invalid formulas: {empty_count}")
    print(f"Mixtures detected: {mixture_count}")
    print(f"Complex nested formulas: {complex_formula_count}")
    print(f"Formulas with unknown elements: {missing_elements_count}")
    print(f"Success rate: {processed_count/len(df)*100:.1f}%")
    print(f"Added {len(features)} new features")
    
    return df

def enhance_csv_file_fixed(input_csv_path, output_csv_path, formula_column='electrolyte'):
    """
    FIXED: Main function to enhance CSV file with robust formula parsing
    """
    print("=" * 70)
    print("FIXED CHEMICAL COMPOSITION FEATURE ENHANCEMENT")
    print("=" * 70)
    print(f"Input file: {input_csv_path}")
    print(f"Output file: {output_csv_path}")
    print(f"Formula column: '{formula_column}'")
    
    # Read CSV
    try:
        df = pd.read_csv(input_csv_path)
        print(f"Successfully loaded CSV with shape: {df.shape}")
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return None
    
    # Validate column
    if formula_column not in df.columns:
        print(f"Error: Column '{formula_column}' not found.")
        print(f"Available columns: {list(df.columns)}")
        return None
    
    # Sample formulas preview
    print(f"\nSample formulas from '{formula_column}' column:")
    sample_formulas = df[formula_column].dropna().head(5)
    for i, formula in enumerate(sample_formulas):
        print(f"  {i+1}. {formula}")
    
    # Test the parser on complex examples
    test_formulas = [
        "((Li2S)0.75(P2S5)0.25)96(P2O5)4",
        "0.75(LiBH4)-0.25Ca(BH4)2",
        "Li3InCl6 (ball milled)"
    ]
    
    print(f"\nTesting parser on complex formulas:")
    for test_formula in test_formulas:
        try:
            composition = parse_mixture_formula(test_formula)
            li_fraction = composition.get('Li', 0) / sum(composition.values()) if composition else 0
            print(f"  '{test_formula}' -> Li fraction: {li_fraction:.3f}")
        except Exception as e:
            print(f"  '{test_formula}' -> Error: {e}")
    
    # Enhance with composition features
    df_enhanced = enhance_composition_features_fixed(df, formula_column)
    
    # Save enhanced dataframe
    try:
        df_enhanced.to_csv(output_csv_path, index=False)
        print(f"\nEnhanced CSV saved successfully to: {output_csv_path}")
    except Exception as e:
        print(f"Error saving CSV: {e}")
        return None
    
    # Feature summary
    print("\n" + "="*70)
    print("NEW FEATURES SUMMARY")
    print("="*70)
    
    new_features = ['avg_electronegativity', 'avg_atomic_mass', 'li_fraction', 
                   'num_elements', 'composition_entropy', 'is_mixture', 'total_atoms']
    
    for feature in new_features:
        if feature in df_enhanced.columns:
            valid_values = df_enhanced[feature].dropna()
            if len(valid_values) > 0:
                if feature == 'is_mixture':
                    mixture_count = valid_values.sum()
                    print(f"{feature}: {mixture_count} mixtures out of {len(valid_values)} formulas")
                elif valid_values.dtype == 'bool':
                    print(f"{feature}: {valid_values.value_counts().to_dict()}")
                else:
                    print(f"{feature}: min={valid_values.min():.3f}, max={valid_values.max():.3f}, mean={valid_values.mean():.3f}")
            else:
                print(f"{feature}: No valid values")
    
    print(f"\nFinal dataframe shape: {df_enhanced.shape}")
    return df_enhanced

# Example usage
if __name__ == "__main__":
    # Test with your problematic formula
    test_formula = "((Li2S)0.75(P2S5)0.25)96(P2O5)4"
    try:
        composition = parse_mixture_formula(test_formula)
        total_atoms = sum(composition.values())
        li_fraction = composition.get('Li', 0) / total_atoms
        print(f"Test: '{test_formula}'")
        print(f"Composition: {composition}")
        print(f"Li fraction: {li_fraction:.3f}")
        print(f"Total atoms: {total_atoms}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Main function call
    enhance_csv_file_fixed("MP_Li_Materials.csv", "MP_Li_Materials_Enhanced.csv", "formula")
