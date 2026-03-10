# Ideas scratchpad

## March 2026 

### Let the Policy Judge Itself: Grounded Self-Distillation Reward Models 

> Current idea we have been iterating on. 

Policy as a Judge (POJ). 


Variants: 

- Self-prompting: llm-as-a-judge but the llm is the policy itself, conditioned on the environment feedback. 
- Self-reward mode: attach a latent reward head that outputs a continious valued scalar (0, 1) as the reward. conditioned on the feedback when doing the reward calculation pass. 
- add in the calibration and l_pred losses to maintain teacher stability: possibly just an ablation. Should start without this. Might not even need it at first. 
- self-reward hack detection: instead of training the latent head to output a reward value, train it to output a hack detection classification 



### Self-Incrimination // Adverserial Reward Hack Detection 

2-3 variants: 

- Self-prompting/distillation judging: condition on the feedback/verified output and prompt self-llm if it was a reward hack or not. 
- self-reward hack detection: instead of training the latent head to output a reward value as in POJ (see above), train it to output a hack detection classification. 
- adverserial two player game setup: exact same setup as in our earlier proposed ARHD system but the labels y^* that we needed come from the two above approaches (self-prompting and self-reward hack model). This is different in that we add include the adverserial objective to the loss. Hence, the minimax game/adverserial. 



### Dynamic reward sparsity for RLVR 

- Right now, RLVR gives 1 bit of information. SDPO proposed token-level distillation which theoretically is the upper bound on the information that can be given. However, sometimes this isn't always a good thing. What if the self-teacher is wrong? or what if it leads to a non-elegant, underoptimized solution? There are many different reasons for why we don't want perfect imitation of the teacher. Sometimes noise is helpful. You don't want to get stuck optimizing in a contained zone of the reward/solution space. Could be better zones in the reward landscape. Thus, ideally, you want some balance between complete sparsity and density. 
- This can help do things like 
 - 1) preserve diversity of solution space, which is measurable by plotting the t-sne of the reward clusters, analyzing pass@k performance in relation to pass@1... should hopefully go up as well
 - 2) smooth out the optimization process 
 - 3) lead to quicker convergence 
 	- theory: if rewards are sparse/1 bit only, then it will take longer for the policy to fizzle into the correct part of the solutuon space. it will neeed to do more exploration, with lots of noise. Thus, varying the bits-per-rollout ratio can potentially lead to quciker convergence to the correct zones. 



### Population-based RLVR for Inducing Exploration 

- Right now, RLVR is done with one seed, one policy but what if you can run n different seeds/variants of the policy at once? They may all end up at different zones in the reward space. This can help improve the diversity of solutions and also induce more exploration. 
- Can potentially help solve the reweight vs. explore debate (i.e., is rlvr just sharpening the distribution already present in the base model or is it exploring fundamentally new solutions). 

