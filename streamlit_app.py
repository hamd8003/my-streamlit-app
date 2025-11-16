import pandas as pd
import numpy as np
import random
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error
import time # To prevent freezing

# --- (Copied from your PART 1) ---
def calculate_calories(row):
    """Calculates daily caloric needs based on user data."""
    if row['Gender'] == 'Male':
        bmr = 88.362 + (13.397 * row['Weight']) + (4.799 * row['Height']) - (5.677 * row['Age'])
    else:
        bmr = 447.593 + (9.247 * row['Weight']) + (3.098 * row['Height']) - (4.330 * row['Age'])
    multiplier = 1.55 
    caloric_need = (bmr * multiplier) * 0.9
    return round(caloric_need)

# --- (Copied from your PART 4, with fixes for calories and variety) ---
def generate_categorized_diet_plan(user_data, food_df, food_type_preference, model_score_column='Predicted_Score_RF'):
    # st.write(f"--- Debug: Generating plan with {model_score_column} ---") # Removed Debug line
    daily_calories = user_data['Daily_Caloric_Needs']
    
    # --- *** MODIFIED: Calorie split changed to 5 meals *** ---
    meal_targets = {
        'Breakfast': daily_calories * 0.25,
        'Pre Lunch': daily_calories * 0.10,
        'Lunch': daily_calories * 0.30,
        'Evening': daily_calories * 0.10,
        'Dinner': daily_calories * 0.25
    }

    # --- *** NEW: Hard limits on item counts per meal *** ---
    meal_item_max = {
        'Breakfast': 3,  # e.g., Milk + 1 Main Dish + 1 Fruit
        'Pre Lunch': 1,  # e.g., 1 Fruit
        'Lunch': 4,      # e.g., Roti + 1 Fruit + 1 Curry + 1 (Rice/Salad)
        'Evening': 1,    # e.g., 1 Fruit or Snack
        'Dinner': 3      # e.g., 1 Main + 1 Side + 1 Fruit
    }
    
    # --- NEW: Get Glucose/HbA1c from the user data ---
    glucose = user_data.get('Glucose', 100) # Default to 100 if not provided
    hba1c = user_data.get('HbA1c', 5.0)     # Default to 5.0 if not provided
    
    # --- *** MODIFIED: WIDENED THRESHOLDS FOR MORE VARIETY *** ---
    if glucose > 180 or hba1c > 7.5:
        # Very high, be very strict
        quantile_threshold = 0.80  # Top 20% of foods (Was 0.90)
        st.warning(f"Warning: High Glucose/HbA1c detected. Recommending from the top 20% strictest food options.")
    elif glucose > 140 or hba1c > 6.5:
        # Elevated, be stricter
        quantile_threshold = 0.70 # Top 30% of foods (Was 0.80)
        st.warning(f"Warning: Elevated Glucose/HbA1c detected. Recommending from the top 30% strictest food options.")
    else:
        # Normal, use a wider pool for variety
        quantile_threshold = 0.60 # Top 40% of foods (Was 0.75)
    
    # --- *** MODIFIED: Smarter food type filtering *** ---
    if food_type_preference.lower() == 'veg':
        # Veg users *only* get Veg items (including fruit)
        available_foods = food_df[food_df['Food_Type'].str.lower() == 'veg'].copy()
    
    elif food_type_preference.lower() == 'non-veg':
        # A Non-Veg user eats Non-Veg dishes, BUT ALSO eats veg staples, fruit, and milk.
        is_non_veg = food_df['Food_Type'].str.lower() == 'non-veg'
        # Define staples that a non-veg user would still eat
        is_staple = food_df['Category'].isin(['Bread/Staple', 'Rice', 'Beverage', 'Condiment'])
        is_fruit = food_df['Meal_Type'].str.contains('fruit', case=False, na=False)
        
        # The available list is Non-Veg items OR Veg staples/fruit/beverages
        available_foods = food_df[is_non_veg | is_staple | is_fruit].copy()
            
    else: # 'Both'
        # 'Both' users can eat everything
        available_foods = food_df.copy()

    full_plan = {}
    used_dishes = set() 

    for meal, target_cals in meal_targets.items():
        # st.write(f"--- Debug: Processing {meal} ---") # Removed Debug line
        
        # --- *** MODIFIED: This list will now store (item, quantity) tuples *** ---
        meal_items_with_qty = []
        meal_calories = 0
        meal_categories = set()

        # --- *** NEW: Get the max item count for this meal *** ---
        max_items = meal_item_max.get(meal, 3) # Default to 3 if not specified

        # --- *** MODIFIED: Include 'fruit' Meal_Type in all meals *** ---
        meal_df = available_foods[
            available_foods['Meal_Type'].str.contains(meal.lower(), case=False, na=False) |
            available_foods['Meal_Type'].str.contains('fruit', case=False, na=False)
        ].copy()


        # --- *** MODIFIED: New logic branch for Snacks vs. Main Meals *** ---
        if meal.lower() in ['pre lunch', 'evening']:
            # --- *** LOGIC FOR SNACKS (Pre Lunch / Evening) *** ---
            
            # Find all valid snacks (Fruit, Beverage, Misc)
            filler_categories = ['Fruit', 'Beverage', 'Miscellaneous']
            snack_options = meal_df[
                (meal_df['Category'].isin(filler_categories)) &
                (~meal_df['DishName'].isin(used_dishes))
            ].sort_values(by=model_score_column, ascending=False)
            
            if not snack_options.empty:
                # Get the top-tier snacks
                score_threshold = snack_options[model_score_column].quantile(quantile_threshold)
                top_snacks = snack_options[snack_options[model_score_column] >= score_threshold]
                
                snack_to_add = None
                if not top_snacks.empty:
                    # --- *** NEW: Randomly sample 1 from the *best* snacks *** ---
                    snack_to_add = top_snacks.sample(n=1).iloc[0]
                elif not snack_options.empty: # Fallback: sample from all
                    snack_to_add = snack_options.sample(n=1).iloc[0]
                
                # Add with quantity 1.0, checking calories
                if snack_to_add is not None and (meal_calories + snack_to_add['Calorieskcal']) <= target_cals * 1.10:
                    meal_items_with_qty.append((snack_to_add, 1.0))
                    used_dishes.add(snack_to_add['DishName'])
                    meal_calories += snack_to_add['Calorieskcal']
        
        else:
            # --- *** LOGIC FOR MAIN MEALS (Breakfast, Lunch, Dinner) *** ---
            
            # --- Pass 0: Mandate Breakfast Staples (Quantity 1.0) ---
            if meal.lower() == 'breakfast' and not meal_df.empty:
                # --- *** MODIFIED: Add 80% probability to milk *** ---
                if random.random() < 0.80: # 80% chance to add milk
                    milk_item_df = meal_df[meal_df['DishName'].str.contains('milk', case=False, na=False)]
                    if not milk_item_df.empty:
                        milk_to_add = milk_item_df.sample(n=1).iloc[0] 
                        if milk_to_add['DishName'] not in used_dishes:
                            meal_items_with_qty.append((milk_to_add, 1.0))
                            used_dishes.add(milk_to_add['DishName'])
                            meal_calories += milk_to_add['Calorieskcal']
                            if pd.notna(milk_to_add['Category']):
                                meal_categories.add(milk_to_add['Category'])
            
            # --- Pass 0: Mandate Lunch Staples (Chapati + Fruit) ---
            if meal.lower() == 'lunch' and not meal_df.empty:
                chapati_item_df = meal_df[meal_df['DishName'].str.contains('chapati|roti', case=False, na=False)]
                if not chapati_item_df.empty:
                    chapati_to_add = chapati_item_df.sample(n=1).iloc[0] 
                    if chapati_to_add['DishName'] not in used_dishes:
                        meal_items_with_qty.append((chapati_to_add, 1.0))
                        used_dishes.add(chapati_to_add['DishName'])
                        meal_calories += chapati_to_add['Calorieskcal']
                        if pd.notna(chapati_to_add['Category']):
                            meal_categories.add(chapati_to_add['Category'])

                if len(meal_items_with_qty) < max_items:
                    fruit_options = available_foods[
                        (available_foods['Meal_Type'].str.contains('fruit', case=False, na=False)) &
                        (~available_foods['DishName'].isin(used_dishes))
                    ].sort_values(by=model_score_column, ascending=False)
                    if not fruit_options.empty:
                        best_fruit = fruit_options.sample(n=1).iloc[0]
                        if (meal_calories + best_fruit['Calorieskcal']) <= target_cals * 1.10:
                            meal_items_with_qty.append((best_fruit, 1.0))
                            used_dishes.add(best_fruit['DishName'])
                            meal_calories += best_fruit['Calorieskcal']
            
            # --- Pass 1: Mandate Fruit for other main meals (Breakfast, Dinner) ---
            if meal.lower() in ['breakfast', 'dinner'] and len(meal_items_with_qty) < max_items:
                has_fruit = any('fruit' in str(item['Meal_Type']).lower() for (item, qty) in meal_items_with_qty)
                if not has_fruit:
                    fruit_options = available_foods[
                        (available_foods['Meal_Type'].str.contains('fruit', case=False, na=False)) &
                        (~available_foods['DishName'].isin(used_dishes))
                    ].sort_values(by=model_score_column, ascending=False)
                    if not fruit_options.empty:
                        best_fruit = fruit_options.sample(n=1).iloc[0]
                        if (meal_calories + best_fruit['Calorieskcal']) <= target_cals * 1.10:
                            meal_items_with_qty.append((best_fruit, 1.0))
                            used_dishes.add(best_fruit['DishName'])
                            meal_calories += best_fruit['Calorieskcal']
            
            # --- *** MODIFIED: Pass 2 (Main Dish Filler) with Randomization *** ---
            slots_remaining = max_items - len(meal_items_with_qty)
            calories_to_fill = target_cals - meal_calories
            
            if slots_remaining > 0 and calories_to_fill > 0 and not meal_df.empty:
                
                filler_categories = ['Curry/Main', 'Dal/Lentil', 'Salad', 'Miscellaneous', 'Rice']
                
                # Get all possible filler options, sorted by score
                filler_options = meal_df[
                    (meal_df['Category'].isin(filler_categories)) &
                    (~meal_df['DishName'].isin(used_dishes))
                ].sort_values(by=model_score_column, ascending=False)
                
                # --- *** MODIFIED: Loop to fill all slots, not just one *** ---
                for _ in range(slots_remaining):
                    
                    # Check if we're done
                    if calories_to_fill <= 0:
                        break
                    
                    # Get the *currently available* top-tier fillers
                    remaining_filler_options = filler_options[~filler_options['DishName'].isin(used_dishes)]
                    if remaining_filler_options.empty:
                        break # No more fillers to add
                        
                    # --- *** ROBUSTNESS FIX: Check if remaining_filler_options is empty before quantile *** ---
                    if remaining_filler_options.empty:
                        break
                        
                    score_threshold_filler = remaining_filler_options[model_score_column].quantile(quantile_threshold)
                    top_fillers = remaining_filler_options[
                        remaining_filler_options[model_score_column] >= score_threshold_filler
                    ]
                    
                    selected_main_dish = None
                    if not top_fillers.empty:
                        selected_main_dish = top_fillers.sample(n=1).iloc[0]
                    elif not remaining_filler_options.empty: # Fallback: sample from any remaining
                        selected_main_dish = remaining_filler_options.sample(n=1).iloc[0]
                    
                    if selected_main_dish is None: # No fillers left
                        break
                        
                    item_calories = selected_main_dish['Calorieskcal']
                    if item_calories <= 0:
                        quantity = 1.0
                    else:
                        # --- *** MODIFIED: Calculate quantity for *this* slot *** ---
                        # Aim to fill a proportional amount of the remaining calories
                        # Or, more simply, just aim for the rest of the calories
                        quantity = calories_to_fill / item_calories
                    
                    quantity = np.round(quantity * 2) / 2 # Round to nearest 0.5
                    quantity = max(0.5, min(3.0, quantity)) # Min 0.5, Max 3 servings
                    
                    # Check if adding this *quantified* item busts the calories
                    added_calories = item_calories * quantity
                    if meal_calories + added_calories > target_cals * 1.15: # Give 15% ceiling
                        quantity = 0.5
                        added_calories = item_calories * quantity
                        if meal_calories + added_calories > target_cals * 1.15:
                            continue
                    
                    meal_items_with_qty.append((selected_main_dish, quantity))
                    used_dishes.add(selected_main_dish['DishName'])
                    
                    # Update loop trackers
                    meal_calories += added_calories
                    slots_remaining -= 1
                    calories_to_fill = target_cals - meal_calories # Recalculate remaining
            
            # --- *** End of Pass 2 *** ---


        # --- *** NEW: Final Total Calculation with Quantities *** ---
        if meal_items_with_qty:
            final_items_list = []
            total_cals = 0
            total_carbs = 0
            total_protein = 0
            total_fibre = 0
            
            for item, qty in meal_items_with_qty:
                # Create a new dict for the item *with* quantity
                item_dict = item.to_dict()
                item_dict['Quantity'] = qty
                final_items_list.append(item_dict)
                
                # Calculate totals by multiplying nutrients by quantity
                total_cals += item['Calorieskcal'] * qty
                total_carbs += item['Carbohydratesg'] * qty
                total_protein += item['Proteing'] * qty
                total_fibre += item['Fibreg'] * qty
            
            totals = {
                'Calories': round(total_cals, 2),
                'Carbohydrates': round(total_carbs, 2),
                'Protein': round(total_protein, 2),
                'Fibre': round(total_fibre, 2)
            }
            full_plan[meal] = {'items': final_items_list, 'totals': totals}
        else:
            full_plan[meal] = {'items': [], 'totals': {k: 0 for k in ['Calories','Carbohydrates','Protein','Fibre']}}

    # st.write("--- Debug: Plan generation complete. ---") # Removed Debug line
    return full_plan


# --- [NEW PART]: CACHED FUNCTION TO LOAD ALL DATA & MODELS ---
@st.cache_data
def load_data_and_train_models():
    # --- PART 2 (Modified): LOAD AND PROCESS FOOD DATA ---
    try:
        food_df = pd.read_csv('food2.csv')
        # --- REMOVED: fruit_df = pd.read_csv('fruit.csv') ---
    except FileNotFoundError as e:
        st.error(f"Error: {e.filename} not found. Make sure it's in the same directory.")
        return None, None, None, None
    except Exception as e:
        st.error(f"An error occurred while loading data: {e}")
        return None, None, None, None


    # --- *** REMOVED: Process and Merge Fruit Data Section *** ---
    
    # 1. Clean main food_df column names
    food_df.columns = food_df.columns.str.replace(r'[\s\(\)µ/]+', '', regex=True).str.strip('')
    
    # --- MORE ROBUST FILLNA (runs on combined df) ---
    for col in food_df.select_dtypes(include=np.number).columns:
        median_val = food_df[col].median()
        if pd.isna(median_val):
            food_df[col] = food_df[col].fillna(0) # Fill with 0 if median is also NaN
        else:
            food_df[col] = food_df[col].fillna(median_val) # Fill with median

    # Categorize (run this *after* merge to catch 'Fruit')
    def get_food_category(dish_name):
        # This is now a simple function again
        dish_name = str(dish_name).lower()
        if any(keyword in dish_name for keyword in ['parantha', 'roti', 'bread', 'naan', 'sandwich']): return 'Bread/Staple'
        if any(keyword in dish_name for keyword in ['dal', 'lentil']): return 'Dal/Lentil'
        if any(keyword in dish_name for keyword in ['curry', 'sabzi', 'masala']) and 'garam' not in dish_name: return 'Curry/Main'
        if any(keyword in dish_name for keyword in ['rice', 'pulao', 'biryani', 'poha']): return 'Rice'
        if any(keyword in dish_name for keyword in ['soup', 'rasam']): return 'Soup'
        if 'salad' in dish_name: return 'Salad'
        if any(keyword in dish_name for keyword in ['pickle', 'achar', 'chutney']): return 'Condiment'
        if any(keyword in dish_name for keyword in ['drink', 'tea', 'coffee', 'juice', 'milk']): return 'Beverage'
        
        # --- *** NEW: Check if it's a fruit by Meal_Type *** ---
        # Note: This is imperfect, as get_food_category doesn't have the Meal_Type.
        # We will rely on the DishName for categorization for now.
        # A better approach would be to categorize *after* loading,
        # but let's see if 'fruit' Meal_Type foods have 'fruit' in their name.
        # For robustness, we will create a 'Fruit' category for any 'fruit' Meal_Type later.
        
        return 'Miscellaneous'

    # Need to apply this row-by-row since 'Category' column now exists
    food_df['Category'] = food_df.apply(lambda row: get_food_category(row['DishName']), axis=1)
    
    food_df['Meal_Type'] = food_df['Meal_Type'].astype(str).str.strip().str.lower()
    food_df['Food_Type'] = food_df['Food_Type'].astype(str).str.strip().str.lower()
    
    # --- *** NEW: Assign 'Fruit' Category based on new Meal_Type *** ---
    # This is the most reliable way to handle your new data structure
    food_df.loc[food_df['Meal_Type'].str.contains('fruit', case=False, na=False), 'Category'] = 'Fruit'
    
    
    food_df = food_df[~food_df['DishName'].str.contains('powder|masala|blend', case=False, na=False)]

    # --- PART 3 (Modified): ML MODEL TRAINING ---
    # st.write("--- Caching: Training all 6 models... (This runs only once) ---") # Removed console log
    
    scaler = StandardScaler()
    nutritional_features = ['Carbohydratesg', 'Proteing', 'Fatsg', 'FreeSugarg', 'Fibreg']
    
    # Make sure all features exist before scaling
    for feature in nutritional_features:
        if feature not in food_df.columns:
            st.error(f"Critical Error: Missing feature '{feature}' in the combined data. Cannot train models.")
            return None, None, None, None
            
    # Filter out rows where any nutritional feature is missing, just in case
    food_df = food_df.dropna(subset=nutritional_features)
    
    food_df_scaled_features = scaler.fit_transform(food_df[nutritional_features])
    food_df_scaled = pd.DataFrame(food_df_scaled_features, columns=nutritional_features, index=food_df.index)

    y = (
        food_df_scaled['Fibreg'] +
        food_df_scaled['Proteing'] -
        food_df_scaled['FreeSugarg'] -
        food_df_scaled['Carbohydratesg'] -
        (food_df_scaled['Fatsg'] * 0.5)
    )
    X = food_df_scaled[nutritional_features]
    
    # Ensure y and X are aligned after any potential dropna
    y = y[X.index]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2) # No random_state for variety
    
    # --- MODIFIED: Added RMSE calculation for all models ---
    
    # Train ALL models
    rf_model = RandomForestRegressor(n_estimators=100)
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    rf_r2 = r2_score(y_test, rf_pred)
    rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))

    svm_model = SVR(kernel='rbf', C=10.0) # Using your best C=10
    svm_model.fit(X_train, y_train)
    svm_pred = svm_model.predict(X_test)
    svm_r2 = r2_score(y_test, svm_pred)
    svm_rmse = np.sqrt(mean_squared_error(y_test, svm_pred))

    gb_model = GradientBoostingRegressor(n_estimators=100)
    gb_model.fit(X_train, y_train)
    gb_pred = gb_model.predict(X_test)
    gb_r2 = r2_score(y_test, gb_pred)
    gb_rmse = np.sqrt(mean_squared_error(y_test, gb_pred))

    xgb_model = XGBRegressor(n_estimators=100, objective='reg:squarederror')
    xgb_model.fit(X_train, y_train)
    xgb_pred = xgb_model.predict(X_test)
    xgb_r2 = r2_score(y_test, xgb_pred)
    xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_pred))
    
    ada_model = AdaBoostRegressor(n_estimators=100) 
    ada_model.fit(X_train, y_train)
    ada_pred = ada_model.predict(X_test)
    ada_r2 = r2_score(y_test, ada_pred)
    ada_rmse = np.sqrt(mean_squared_error(y_test, ada_pred))
    
    mlp_model = MLPRegressor(hidden_layer_sizes=(100,), max_iter=500, early_stopping=True)
    mlp_model.fit(X_train, y_train)
    mlp_pred = mlp_model.predict(X_test)
    mlp_r2 = r2_score(y_test, mlp_pred)
    mlp_rmse = np.sqrt(mean_squared_error(y_test, mlp_pred))
    
    # Add predictions for ALL models to the food_df
    food_df['Predicted_Score_RF'] = rf_model.predict(X)
    food_df['Predicted_Score_SVM'] = svm_model.predict(X)
    food_df['Predicted_Score_GB'] = gb_model.predict(X)
    food_df['Predicted_Score_XGB'] = xgb_model.predict(X)
    food_df['Predicted_Score_ADA'] = ada_model.predict(X)
    food_df['Predicted_Score_MLP'] = mlp_model.predict(X)
    
    # Store models and their R2 scores in a dictionary
    model_scores = {
        "Random Forest": rf_r2,
        "SVM (SVR)": svm_r2,
        "Gradient Boosting": gb_r2,
        "XGBoost": xgb_r2,
        "AdaBoost": ada_r2,
        "MLP (Neural Net)": mlp_r2,
    }

    # --- NEW: Store RMSE scores in a new dictionary ---
    model_rmse_scores = {
        "Random Forest": rf_rmse,
        "SVM (SVR)": svm_rmse,
        "Gradient Boosting": gb_rmse,
        "XGBoost": xgb_rmse,
        "AdaBoost": ada_rmse,
        "MLP (Neural Net)": mlp_rmse,
    }

    # Map for the GUI to use
    model_column_map = {
        "Random Forest": "Predicted_Score_RF",
        "SVM (SVR)": "Predicted_Score_SVM",
        "Gradient Boosting": "Predicted_Score_GB",
        "XGBoost": "Predicted_Score_XGB",
        "AdaBoost": "Predicted_Score_ADA",
        "MLP (Neural Net)": "Predicted_Score_MLP",
    }
    
    # st.write("--- Caching: Model training complete! ---") # Removed console log
    
    return food_df, model_scores, model_rmse_scores, model_column_map

# --- [NEW PART]: STREAMLIT APP UI ---

st.set_page_config(layout="wide")
st.title("Diabetic Patient Food Recommendation System")

# --- *** NEW: Add CSS to reduce metric font size *** ---
st.markdown("""
<style>
[data-testid="stMetricValue"] {
    font-size: 1.75rem;
}
</style>
""", unsafe_allow_html=True)

# --- *** NEW: Added caption about serving size *** ---
st.caption("Nutritional information for all dishes is based on a standard serving (e.g., 100g for main dishes, 1 bowl for soup, 1 piece for bread).")

# Load all data and models (this will be cached)
# --- MODIFIED: Added model_rmse_scores to the return values ---
with st.spinner("Loading models and all food data... (This is a one-time operation)"):
    food_df, model_scores, model_rmse_scores, model_column_map = load_data_and_train_models()

# --- Sidebar for User Input ---
st.sidebar.header("Enter Patient Details")

# --- MODIFIED: Added Glucose and HbA1c ---
age = st.sidebar.number_input("Age", min_value=18, max_value=100, value=55)
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
weight = st.sidebar.number_input("Weight (kg)", min_value=40.0, max_value=150.0, value=70.0)
height = st.sidebar.number_input("Height (cm)", min_value=120.0, max_value=220.0, value=170.0)
glucose = st.sidebar.number_input("Current Glucose (mg/dL)", min_value=70, max_value=400, value=120)
hba1c = st.sidebar.number_input("HbA1c (%)", min_value=4.0, max_value=15.0, value=6.0, step=0.1)
diet_pref = st.sidebar.selectbox("Dietary Preference", ["Veg", "Non-Veg", "Both"])

st.sidebar.header("Select Model")

# Check if model_scores is not None before finding the best model
if model_scores and model_rmse_scores:
    best_model_name = max(model_scores, key=model_scores.get)
    best_model_index = list(model_scores.keys()).index(best_model_name)
    
    model_choice = st.sidebar.selectbox(
        "Choose a Model (or use the best)",
        options=list(model_scores.keys()),
        index=best_model_index
    )
    
    # --- *** MODIFIED: REMOVED ALL METRICS FROM SIDEBAR *** ---
    # (The expander and metrics have been deleted)

else:
    st.sidebar.error("Models could not be trained.")
    model_choice = None

# --- Main App Body ---
submit_button = st.sidebar.button("Generate Diet Plan")

if submit_button and food_df is not None and model_choice is not None:
    # 1. Create the user_data from inputs
    # --- MODIFIED: Added Glucose and HbA1c ---
    user_data = pd.Series({
        'Age': age,
        'Gender': gender,
        'Weight': weight,
        'Height': height,
        'Glucose': glucose,
        'HbA1c': hba1c
    })
    
    # 2. Calculate calories
    user_data['Daily_Caloric_Needs'] = calculate_calories(user_data)
    
    # 3. Get the correct score column name from the user's choice
    selected_score_column = model_column_map[model_choice]

    st.header(f"Generated Plan using: {model_choice}")
    st.subheader(f"Target Daily Calories: ~{user_data['Daily_Caloric_Needs']} kcal")
    
    # 4. Run the recommendation
    with st.spinner("Building your personalized meal plan..."):
        plan = generate_categorized_diet_plan(user_data, food_df, diet_pref, selected_score_column)

    # 5. Display the results
    grand_totals = {k: 0 for k in ['Calories', 'Carbohydrates', 'Protein', 'Fibre']}
    
    # --- *** MODIFIED: Changed to 5-meal loop *** ---
    for meal in ['Breakfast', 'Pre Lunch', 'Lunch', 'Evening', 'Dinner']:
        data = plan.get(meal)
        if not data:
            continue
            
        # --- MODIFIED: Removed hyphens ---
        st.subheader(f"{meal.upper()}")
        
        if not data['items']:
            st.write("No items selected for this meal.")
        else:
            # Display items as a table
            items_df = pd.DataFrame(data['items'])
            
            # --- *** MODIFIED: Removed 'Category' from display *** ---
            display_cols = ['DishName', 'Quantity', selected_score_column, 'Calorieskcal', 'Carbohydratesg', 'Proteing', 'Fibreg']
            
            # Filter out columns that might be missing if something went wrong
            display_cols = [col for col in display_cols if col in items_df.columns]
            
            # Rename score col for clarity
            items_df_display = items_df[display_cols].rename(columns={
                selected_score_column: "Suitability Score",
                'Calorieskcal': 'Cals (per 1)',
                'Carbohydratesg': 'Carbs (per 1)',
                'Proteing': 'Protein (per 1)',
                'Fibreg': 'Fibre (per 1)'
                })
            
            st.dataframe(items_df_display.style.format({'Suitability Score': "{:.2f}", 'Quantity': "{:.1f}"}))

            # --- MODIFIED: Replaced st.json with st.metric ---
            st.markdown(f"**Meal Totals (with Quantity):**")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric(label="Calories", value=f"{data['totals']['Calories']} kcal")
            with col2:
                st.metric(label="Carbs", value=f"{data['totals']['Carbohydrates']} g")
            with col3:
                st.metric(label="Protein", value=f"{data['totals']['Protein']} g")
            with col4:
                st.metric(label="Fibre", value=f"{data['totals']['Fibre']} g")
            
            # Add to grand totals
            for nutrient, value in data['totals'].items():
                grand_totals[nutrient] += value
    
    st.header("Total Nutrition for the Day")
    # --- MODIFIED: Replaced st.json with st.metric ---
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(label="Total Calories", value=f"{round(grand_totals['Calories'], 2)} kcal")
    with col2:
        st.metric(label="Total Carbs", value=f"{round(grand_totals['Carbohydrates'], 2)} g")
    with col3:
        st.metric(label="Total Protein", value=f"{round(grand_totals['Protein'], 2)} g")
    with col4:
        st.metric(label="Total Fibre", value=f"{round(grand_totals['Fibre'], 2)} g")
    
    # --- MODIFIED: Added st.metric for Target vs Achieved ---
    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="Target Calories", value=f"{user_data['Daily_Caloric_Needs']} kcal")
    with col2:
        # Add a delta to show the difference
        delta_cal = round(grand_totals['Calories'] - user_data['Daily_Caloric_Needs'], 2)
        st.metric(label="Achieved Calories", value=f"{round(grand_totals['Calories'], 2)} kcal", delta=f"{delta_cal} kcal")


elif not (food_df is not None and model_scores is not None and model_column_map is not None):
     # --- MODIFIED: Updated error message ---
     st.error("Data or models could not be loaded. Please check your 'food2.csv' file and restart the app.")
else:
    st.info("Please enter patient details and click 'Generate Diet Plan' in the sidebar.")
