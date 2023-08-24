# Task Decomposition in Non-Static Deep Reinforcement Learning Robotic Environments

![Complete_task](https://github.com/nikolaos1997/Neural-Robotics/assets/103045738/a8a9a811-f829-4d4f-9dba-76d84e35d8b8)

# Abstract
Deep Reinforcement Learning in robotic applications has shown promising outcomes in goal-based tasks over classical control methods. Yet, conventional intelligent robotic set-ups often overlook real-world scenarios in simulated environments, limiting deployment and applicability. Our research aimed to address this by designing and solving an object manipulation robotic task with moving components, utilizing biologically-inspired concepts as temporal integration of sub-decision, and attention mechanisms. 

Based on the 'Fetch' robot hand from OpenAI gym, our designed task was to pick up a cube from one moving platform and place it onto another moving target platform. We decompose the end-end task into three interconnected and sequential sub-tasks. We train the robot agent to complete the sub-tasks with the Soft Actor Critic framework based on sequences of environmental states and past actions. We further investigated the capabilities of self-attention module in the employed neural networks.

Incorporating self-attention demonstrated significantly high success rates across all sub-tasks during training phase, while task decomposition was crucial for the successful completion of the end-end task. When utilizing only linear layers in our neural networks yielded much lower success rates in each sub-task, showcasing the capabilities of self-attention module in real world scenarios with moving objects. The adaptability of the self-attention module was further proved when exposed to varying platform speeds, different from the speed that was used during the training phases, maintaining high performance. However, self-attention module was found to be prune to input state sequence feature perturbations, due to the inner complicated calculations.

Our study investigates the application of biologically-inspired mechanisms in dynamic robotic environments, bridging the gap between simulation and real world challenges. The insights obtained serve as foundational knowledge for real world robotic applications, paving also at promising future research directions in continual and meta-learning.
