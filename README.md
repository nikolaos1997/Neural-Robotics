# Improving Learning in a Decomposed Continuous Control Task with Self-Attention

![Complete_task](https://github.com/nikolaos1997/Neural-Robotics/assets/103045738/a8a9a811-f829-4d4f-9dba-76d84e35d8b8)

Deep Reinforcement Learning in robotic applications has shown promis-
ing outcomes in goal-based tasks over classical control methods. Yet, standard robot
simulation set-ups used in the Reinforcement Learning literature do not typically in-
clude moving targets for the control policy to achieve the goal, limiting the complexity
to a non-realistic problem. Our research aims to address this by designing and solving
an object manipulation robotic task with moving components, utilizing biologically-
inspired concepts as temporal integration of sub-tasks, and attention mechanisms. Based on the ’Fetch’ robot hand from OpenAI gym, our designed task was
to pick up a cube from one moving platform and place it onto another moving target
platform. We decompose the end-end task into three interconnected and sequential
sub-tasks. We train the robot agent to complete the sub-tasks with the Soft Actor Critic
Reinforcement Learning framework based on sequences of environmental states and
past actions. Furthermore, we investigated the capabilities of self-attention method in
the employed neural networks. Incorporating self-attention demonstrated significantly high success rates across
all sub-tasks during training phase, while task decomposition was crucial for the suc-
cessful completion of the end-end task. Utilizing only linear layers in our neural net-
works yielded much lower success rates in each sub-task, illustrating the capabilities
of self-attention module in real world scenarios with moving objects. The adaptability
of the self-attention module in continuous control was shown when exposed to vary-
ing platform speeds, different from the speed that was used during the training phases,
maintaining high performance. However, self-attention module was found susceptible
to input state sequence feature perturbations, due to the inner complicated calculations. Our study investigates the application of biologically-inspired mechanisms
in dynamic robotic environments, bridging the gap between simulation and real world
challenges. The insights obtained serve as foundational knowledge for real world robotic
applications, paving also at promising future research directions
