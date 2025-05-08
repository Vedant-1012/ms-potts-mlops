# tools.py

def meal_logging(query, user_context):
    """Simulate extracting and logging meal information."""
    # In real version, you would parse the query with LLM.
    # For now, just echo back the meal entry.
    return {
        "reasoning": f"Logged meal based on user input: '{query}'.",
        "final_answer": f"Meal '{query}' successfully logged for user {user_context.get('name', 'Unknown')}.",
        "detected_intent": "Meal-Logging",
        "context_used": "User meal logging simulation."
    }

def meal_planning(user_context):
    """Simulate generating a simple meal plan."""
    goal = user_context.get('goal', 'healthy eating')
    allergies = user_context.get('allergies', '')
    diet_restrictions = f"Avoid these ingredients: {allergies}" if allergies else "No specific restrictions."

    plan = f"""
    Here is a basic 3-day meal plan for your goal: {goal}.

    Day 1:
      - Breakfast: Oatmeal with berries
      - Lunch: Grilled chicken salad
      - Dinner: Steamed salmon with quinoa

    Day 2:
      - Breakfast: Greek yogurt with nuts
      - Lunch: Vegetable stir-fry with tofu
      - Dinner: Lentil soup with whole grain bread

    Day 3:
      - Breakfast: Smoothie with spinach and banana
      - Lunch: Chickpea salad
      - Dinner: Grilled turkey burger with sweet potatoes

    {diet_restrictions}
    """
    return plan
