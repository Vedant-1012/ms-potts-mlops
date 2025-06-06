from ms_potts.tools import meal_logging, meal_planning


def test_meal_logging():
    query = "Grilled chicken wrap with veggies"
    user_context = {"name": "Vedant"}

    result = meal_logging(query, user_context)

    assert isinstance(result, dict)
    assert result["detected_intent"] == "Meal-Logging"
    assert "Grilled chicken wrap" in result["final_answer"]
    assert "Vedant" in result["final_answer"]
    assert "Logged meal" in result["reasoning"]


def test_meal_planning_with_goal_and_allergies():
    user_context = {"goal": "muscle gain", "allergies": "nuts"}

    result = meal_planning(user_context)

    assert isinstance(result, str)
    assert "muscle gain" in result
    assert "Avoid these ingredients: nuts" in result
    assert "Day 1:" in result


def test_meal_planning_without_allergies():
    user_context = {"goal": "weight loss"}

    result = meal_planning(user_context)

    assert "No specific restrictions" in result
    assert "weight loss" in result
