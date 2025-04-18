import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

def assess_brain_score(answers):
    answer = ctrl.Antecedent(np.arange(0, 5, 1), 'answer')
    score = ctrl.Consequent(np.arange(0, 101, 1), 'score')

    # Define membership functions
    answer['Never'] = fuzz.trimf(answer.universe, [0, 0, 1])
    answer['Almost Never'] = fuzz.trimf(answer.universe, [0, 1, 2])
    answer['Sometimes'] = fuzz.trimf(answer.universe, [1, 2, 3])
    answer['Fairly Often'] = fuzz.trimf(answer.universe, [2, 3, 4])
    answer['Very Often'] = fuzz.trimf(answer.universe, [3, 4, 4])

    score['Poor'] = fuzz.trimf(score.universe, [0, 0, 50])
    score['Good'] = fuzz.trimf(score.universe, [0, 50, 100])

    rule1 = ctrl.Rule(answer['Never'] | answer['Almost Never'] | answer['Sometimes'], score['Good'])
    rule2 = ctrl.Rule(answer['Fairly Often'] | answer['Very Often'], score['Poor'])

    health_score_ctrl = ctrl.ControlSystem([rule1, rule2])
    health_score = ctrl.ControlSystemSimulation(health_score_ctrl)

    for ans in answers:
        health_score.input['answer'] = ans
        health_score.compute()

    return health_score.output['score']
