# Configuration Discription
All settings are stored in the folder `configs/exprienments`, if you want to train the model using your own settings, please modify `configs/dllsoa_template.yaml`.
## 数据集
- MNIST
- CIFAR10
## 完全不通信
1. 本地训练不通信
## 最好的情况
2. 子载波不受限，功率不受限，选择策略random（不受限的情况下，选啥都是一样的），聚合前先加噪声(dpsgd)
3. 子载波不受限，功率不受限，选择策略random（不受限的情况下，选啥都是一样的），聚合后才加噪声(dllsoa)
## 子载波受限的情况
4. 子载波受限1，功率不受限，选择random，$\beta=1.0$ 无noise
5. 子载波受限1，功率不受限，选择weight，$\beta=1.0$ 无noise
6. 子载波受限1，功率不受限，选择grad，$\beta=1.0$ 无noise
7. 子载波受限2，功率不受限，选择random，$\beta=1.0$ 无noise
8. 子载波受限2，功率不受限，选择weight，$\beta=1.0$ 无noise
9. 子载波受限2，功率不受限，选择grad，$\beta=1.0$ 无noise
10. 子载波受限3，功率不受限，选择random，$\beta=1.0$ 无noise
11. 子载波受限3，功率不受限，选择weight，$\beta=1.0$ 无noise
12. 子载波受限3，功率不受限，选择grad，$\beta=1.0$ 无noise
## 功率受限下
13. 子载波不受限，功率受限，b的计算使用avg,alpha用eq5,选择random，$\beta=0.8$+noise
14. 子载波不受限，功率受限，b的计算使用avg,alpha用eq6,选择random，$\beta=0.8$+noise
15. 子载波不受限，功率受限，b的计算使用eq3,alpha用eq5,选择random，$\beta=0.8$+noise
16. 子载波不受限，功率受限，b的计算使用eq3,alpha用eq6,选择random，$\beta=0.8$+noise

## 都受限的情况下
17. 子载波受限2，功率受限，b的计算使用eq3,alpha用eq5,选择random, $\beta=0.5$+noise
18. 子载波受限2，功率受限，b的计算使用eq3,alpha用eq5,选择weight, $\beta=0.5$+noise
19. 子载波受限2，功率受限，b的计算使用eq3,alpha用eq5,选择grad, $\beta=0.5$+noise

20. 子载波受限2，功率受限，b的计算使用eq3,alpha用eq6,选择random, $\beta=0.5$+noise
21. 子载波受限2，功率受限，b的计算使用eq3,alpha用eq6,选择weight, $\beta=0.5$+noise
22. 子载波受限2，功率受限，b的计算使用eq3,alpha用eq6,选择grad, $\beta=0.5$+noise

23. 子载波受限2，功率受限，b的计算使用eq3,alpha用eq5,选择random, $\beta=1.0$+noise
24. 子载波受限2，功率受限，b的计算使用eq3,alpha用eq5,选择weight, $\beta=1.0$+noise
25. 子载波受限2，功率受限，b的计算使用eq3,alpha用eq5,选择grad, $\beta=1.0$+noise

26. 子载波受限2，功率受限，b的计算使用eq3,alpha用eq6,选择random, $\beta=1.0$+noise
27. 子载波受限2，功率受限，b的计算使用eq3,alpha用eq6,选择weight, $\beta=1.0$+noise
28. 子载波受限2，功率受限，b的计算使用eq3,alpha用eq6,选择grad, $\beta=1.0$+noise

29. 子载波受限2，功率受限，b的计算使用eq3,alpha用eq5,选择random, $\beta=1.5$+noise
30. 子载波受限2，功率受限，b的计算使用eq3,alpha用eq5,选择weight, $\beta=1.5$+noise
31. 子载波受限2，功率受限，b的计算使用eq3,alpha用eq5,选择grad, $\beta=1.5$+noise

32. 子载波受限2，功率受限，b的计算使用eq3,alpha用eq6,选择random, $\beta=1.5$+noise
33. 子载波受限2，功率受限，b的计算使用eq3,alpha用eq6,选择weight, $\beta=1.5$+noise
34. 子载波受限2，功率受限，b的计算使用eq3,alpha用eq6,选择grad, $\beta=1.5$+noise

## 无噪声的情况
35. 子载波不受限，功率不受限，选择策略random（不受限的情况下，选啥都是一样的），聚合前先加噪声(dpsgd)，噪声的$\sigma=0$ 
36. 子载波不受限，功率不受限，选择策略random（不受限的情况下，选啥都是一样的），聚合后才加噪声(dllsoa)，噪声的$\sigma=0$ 