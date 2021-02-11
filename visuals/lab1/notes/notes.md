# Lab 1 notes (ušporko)

Excel sheet with some results:
<https://docs.google.com/spreadsheets/d/1fv7UyF2UmWiX7Lin4GtoRyjfZfZsXrGy9lFEeq9bXfQ/edit#gid=0>

## 4 - pt_logreg

logreg vs pt_logreg - 1 - no weight decay - same
![logreg vs pt_logreg - 1 - no weight decay - same --- e to to.png](../lab1%20-%20logreg%20vs%20pt_logreg%20-%201%20-%20no%20weight%20decay%20-%20same%20---%20e%20to%20to.png)

logreg vs pt_logreg - 2 - with weight decay - not same cause initial pt version applied L2 to bias
![logreg vs pt_logreg - 2 - with weight decay - not same cause initial pt version applied L2 to bias.png](../lab1%20-%20logreg%20vs%20pt_logreg%20-%202%20-%20with%20weight%20decay%20-%20not%20same%20cause%20initial%20pt%20version%20applied%20L2%20to%20bias.png)

logreg vs pt_logreg - 3 - with weight decay - same
![logreg vs pt_logreg - 3 - with weight decay - same.png](../lab1%20-%20logreg%20vs%20pt_logreg%20-%203%20-%20with%20weight%20decay%20-%20same.png)

pt_logreg eta=0.1 lambda=1 epochs=30000 --- too much regularization
![pt_logreg eta=0.1 lambda=1 epochs=30000 --- too much regularization.png](../lab1%20-%20pt_logreg%20eta=0.1%20lambda=1%20epochs=30000%20---%20too%20much%20regularization.png)

pt_logreg eta=10 lambda=0 epochs=30000 --- eta too high, divergence
![pt_logreg eta=10 lambda=0 epochs=30000 --- eta too high, divergence.png](../lab1%20-%20pt_logreg%20eta=10%20lambda=0%20epochs=30000%20---%20eta%20too%20high,%20divergence.png)

## 5 - pt_deep - shallow vs deep - relu vs sigmoid

Napisao sam PTDeep i PTDeep2, prvi je "na ruke" dok potonji koristi gotove PyTorch funkcionalnosti (torch.nn.Linear, torch.nn.Sequential). Jedina je razlika između njih što PTDeep2 koristi kaiming_uniform_ za inicijalizaciju parametara težina i pristranosti, dok PTDeep koristi inicijalizira težine sa N(0,1), a pristranosti sa nulama. Nakon što sam podesio da se i PTDeep2 inicijalizira na isti način kao i PTDeep, davali su identične rezultate na MNIST skupu podataka. Primjetio sam da je PTDeep2 mjerljivo brži od PTDeep i da uz kaiming_uniform_ inicijalizaciju daje brže bolje rezultate.

### Stats

Parametri su isti za sve. "Hidden layers" govori koji je od ovih:
0 --> [D,C]
1 --> [D,10,C]
2 --> [D,10,20,C]

A datasetovi su ovi:
```py
    ...
    np.random.seed(100)
    ...

    @staticmethod
    def load_dataset(id):
        if id == 1:
            return sample_gauss_2d(3, 100)
        elif id == 2:
            return sample_gmm_2d(4, 2, 40)
        elif id == 3:
            return sample_gmm_2d(6, 2, 10)
        else:
            raise RuntimeError(f"Dataset with id {id} is not supported")
```

#### RELU

![RELU results](2021-02-02-21-56-44.png)

#### SIGMOID

![SIGMOID results](2021-02-02-21-59-01.png)

Tko je bio bolji? Za ove parametre, ReLU je imao bolje rezultate na sva tri dataseta. Probao sam i bez whiteninga za sigmoidu na zadnjem datasetu, ali je ReLU i dalje bolji. Mislim da je to do datasetova i toga da se na ovim datasetovima ReLu uspio bolje prenaučiti.

TODO Uputa kaze: "Sigmoida bi za ovakve male probleme zbog neprekidnosti trebala postići bolje rezultate od zglobnice" Nisam dobio takve rezultate.

Da se primjetiti da bez hidden layera nije bitna funkcija aktivacije, sto je ocekivano.

### Decision boundaries

#### RELU

![relu --- pt_deep w=1 v=2  act=relu eta=0.1 lambda=0.0001 epochs=30000.png](../lab1%20-%20relu%20---%20pt_deep%20w=1%20v=2%20%20act=relu%20eta=0.1%20lambda=0.0001%20epochs=30000.png)

#### SIGMOID

![sigmoid --- pt_deep w=1 v=2  act=sigmoid eta=0.1 lambda=0.0001 epochs=30000.png](../lab1%20-%20sigmoid%20---%20pt_deep%20w=1%20v=2%20%20act=sigmoid%20eta=0.1%20lambda=0.0001%20epochs=30000.png)

Moze se na granicama primjetiti je li ReLU ili je Sigmoida koristena. ReLU je linearan po dijelovima, sigmoida ima krivulje

### Svi rezultati

#### ReLU

```js
1
'(recall_i, precision_i)': [(0.97,
                              0.9603960396039604),
                             (0.96,
                              0.9795918367346939),
                             (1.0,
                              0.9900990099009901)],
 'accuracy': 0.9766666666666667,
 'confusion matrix': array([[ 97,   2,   1],
       [  4,  96,   0],
       [  0,   0, 100]]),
 'train_loss': tensor(0.0824, dtype=torch.float64, grad_fn=<AddBackward0>)
2
'(recall_i, precision_i)': [(0.98,
                              0.9702970297029703),
                             (0.97,
                              0.9797979797979798),
                             (1.0, 1.0)],
 'accuracy': 0.9833333333333333,
 'confusion matrix': array([[ 98,   2,   0],
       [  3,  97,   0],
       [  0,   0, 100]]),
 'train_loss': tensor(0.0514, dtype=torch.float64, grad_fn=<AddBackward0>)
3
'(recall_i, precision_i)': [(0.99,
                              0.99),
                             (0.99,
                              0.99),
                             (1.0, 1.0)],
 'accuracy': 0.9933333333333333,
 'confusion matrix': array([[ 99,   1,   0],
       [  1,  99,   0],
       [  0,   0, 100]]),
 'train_loss': tensor(0.0235, dtype=torch.float64, grad_fn=<AddBackward0>)
4
'(recall_i, precision_i)': [(0.5875,
                              0.7121212121212122),
                             (0.7625,
                              0.648936170212766)],
 'accuracy': 0.675,
 'confusion matrix': array([[47, 33],
       [19, 61]]),
 'train_loss': tensor(0.4415, dtype=torch.float64, grad_fn=<AddBackward0>)
5
'(recall_i, precision_i)': [(0.9625,
                              0.9625),
                             (0.9625,
                              0.9625)],
 'accuracy': 0.9625,
 'confusion matrix': array([[77,  3],
       [ 3, 77]]),
 'train_loss': tensor(0.1277, dtype=torch.float64, grad_fn=<AddBackward0>)
6
'(recall_i, precision_i)': [(0.9875,
                              0.9634146341463414),
                             (0.9625,
                              0.9871794871794872)],
 'accuracy': 0.975,
 'confusion matrix': array([[79,  1],
       [ 3, 77]]),
 'train_loss': tensor(0.1032, dtype=torch.float64, grad_fn=<AddBackward0>)
7
'(recall_i, precision_i)': [(0.5666666666666667,
                              0.6071428571428571),
                             (0.6333333333333333,
                              0.59375)],
 'accuracy': 0.6,
 'confusion matrix': array([[17, 13],
       [11, 19]]),
 'train_loss': tensor(0.6682, dtype=torch.float64, grad_fn=<AddBackward0>)
8
'(recall_i, precision_i)': [(1.0,
                              0.967741935483871),
                             (0.9666666666666667,
                              1.0)],
 'accuracy': 0.9833333333333333,
 'confusion matrix': array([[30,  0],
       [ 1, 29]]),
 'train_loss': tensor(0.1020, dtype=torch.float64, grad_fn=<AddBackward0>)
9
'(recall_i, precision_i)': [(1.0, 1.0),
                             (1.0, 1.0)],
 'accuracy': 1.0,
 'confusion matrix': array([[30,  0],
       [ 0, 30]]),
 'train_loss': tensor(0.0246, dtype=torch.float64, grad_fn=<AddBackward0>)
```

#### Sigmoid

```js
1
'(recall_i, precision_i)': [(0.97,
                              0.9603960396039604),
                             (0.96,
                              0.9795918367346939),
                             (1.0,
                              0.9900990099009901)],
 'accuracy': 0.9766666666666667,
 'confusion matrix': array([[ 97,   2,   1],
       [  4,  96,   0],
       [  0,   0, 100]]),
 'train_loss': tensor(0.0824, dtype=torch.float64, grad_fn=<AddBackward0>)
2
'(recall_i, precision_i)': [(0.98,
                              0.9607843137254902),
                             (0.96,
                              0.9896907216494846),
                             (1.0,
                              0.9900990099009901)],
 'accuracy': 0.98,
 'confusion matrix': array([[ 98,   1,   1],
       [  4,  96,   0],
       [  0,   0, 100]]),
 'train_loss': tensor(0.0778, dtype=torch.float64, grad_fn=<AddBackward0>)
3
'(recall_i, precision_i)': [(0.99,
                              0.9705882352941176),
                             (0.97,
                              0.9897959183673469),
                             (1.0, 1.0)],
 'accuracy': 0.9866666666666667,
 'confusion matrix': array([[ 99,   1,   0],
       [  3,  97,   0],
       [  0,   0, 100]]),
 'train_loss': tensor(0.0796, dtype=torch.float64, grad_fn=<AddBackward0>)
4
'(recall_i, precision_i)': [(0.5875,
                              0.7121212121212122),
                             (0.7625,
                              0.648936170212766)],
 'accuracy': 0.675,
 'confusion matrix': array([[47, 33],
       [19, 61]]),
 'train_loss': tensor(0.4415, dtype=torch.float64, grad_fn=<AddBackward0>)
5
'(recall_i, precision_i)': [(0.9625,
                              0.927710843373494),
                             (0.925,
                              0.961038961038961)],
 'accuracy': 0.94375,
 'confusion matrix': array([[77,  3],
       [ 6, 74]]),
 'train_loss': tensor(0.1766, dtype=torch.float64, grad_fn=<AddBackward0>)
6
'(recall_i, precision_i)': [(0.9625,
                              0.9390243902439024),
                             (0.9375,
                              0.9615384615384616)],
 'accuracy': 0.95,
 'confusion matrix': array([[77,  3],
       [ 5, 75]]),
 'train_loss': tensor(0.1612, dtype=torch.float64, grad_fn=<AddBackward0>)
7
'(recall_i, precision_i)': [(0.5666666666666667,
                              0.6071428571428571),
                             (0.6333333333333333,
                              0.59375)],
 'accuracy': 0.6,
 'confusion matrix': array([[17, 13],
       [11, 19]]),
 'train_loss': tensor(0.6682, dtype=torch.float64, grad_fn=<AddBackward0>)
8
'(recall_i, precision_i)': [(1.0,
                              0.967741935483871),
                             (0.9666666666666667,
                              1.0)],
 'accuracy': 0.9833333333333333,
 'confusion matrix': array([[30,  0],
       [ 1, 29]]),
 'train_loss': tensor(0.1745, dtype=torch.float64, grad_fn=<AddBackward0>)
9
'(recall_i, precision_i)': [(1.0,
                              0.967741935483871),
                             (0.9666666666666667,
                              1.0)],
 'accuracy': 0.9833333333333333,
 'confusion matrix': array([[30,  0],
       [ 1, 29]]),
 'train_loss': tensor(0.1190, dtype=torch.float64, grad_fn=<AddBackward0>)
```

## 6 - ksvm_wrap

### Usporedite performansu modela koje implementiraju razredi PTDeep i KSVMWrap na većem broju slučajnih skupova podataka. Koje su prednosti i nedostatci njihovih funkcija gubitka? Koji od dvaju postupaka daje bolju garantiranu performansu? Koji od postupaka može primiti veći broj parametara? Koji bi od postupaka bio prikladniji za 2D podatke uzorkovane iz mješavine Gaussovi distribucija?

Pokrenuo sam na tri skupa podataka, na istim kao i prethodno pt_deep. Izabrani rezultati:
![Izabrani rezultati za ksvm_wrap nad trima datasetovima](2021-02-03-10-59-42.png)

U skromnom broju eksperimenata koje sam pokrenuo i u kojima postoji samo skup za treniranje, ksvm_wrap i pt_deep su uspijevali postici podjednako visok accuracy (ali se na decizisjim plohama moze vidjeti da je za to kriva prenaucenost kojom nisam upravljao). pt_deep je postigao nizi gubitak mjeren kao nLL. Intuitivno je to jasno, pt_deep direktno minimizira takvu funkciju gubitka, dok svm gura podatke dalje od decizijske plohe.

Jedna prednost nLL kao funckije gubitka jest da se radi ML procjena koja ima probabilisticko objasnjenje. Jedan nedostatak ML procjene jest jednostavna prenaucenost, osobito u nedostatku podatka.

Jedna prednost SVMove funkcije gubitka koja maximizira udaljenost podataka od decizijske plohe jest mogucnost stvaranja intuitivnih i realisticnih decizijskih ploha, ako je skup podataka za to prikladan (nije prevelik i nema previse noisea). TODO jedan nedostatak :-)

Koji od dvaju postupaka daje bolju garantiranu performansu? No free lunch. Na testiranim skupovima s malo podataka je SVM mnogo brzi, a oboje su se uspjeli prenauciti. pt_deep je imao ocekivano nizi gubitak. Mislim da je na temelju eksperimenata hrabro zakljuciti vise od toga. Doduše, ako smijemo samo jedan model trenirati, SVM obećaje više za ovaj skup podataka jer je pt_deep osjetljiv na inicijalicaciju.

pt_deep je model koji moze imati proizvoljan broj parametara, odnosno proizvoljnu dubinu i sirinu, dok SVM ima nekolicinu hiperparametara poput 1. c 2. gamma 3. rbf/linear/poly 4.ovr vs ovo itd. SVM ne mijenja broj ucenih parametara kao pt_deep koji njima direktno upravlja pomocu hiperparametara (dubina i sirina svakog sloja).

Koji bi od postupaka bio prikladniji za 2D podatke uzorkovane iz mješavine Gaussovi distribucija? Mislim da je RBF SVM dao intuitivnije decizijske plohe od relu i sigmoid MLPova. Ali korišteni skupovi podatala nisu imali mnogo šuma pa je RBFu posao bio olakšan.

Pripadni grafovi rezultata su u nastavku.

#### Hard classification

![lab 1 - 6 - ksvm_wrap 1.png](../6_ksvm_wrap_2d/1.png)

#### Probability for most likely class

![lab 1 - 6 - ksvm_wrap 2.png](../6_ksvm_wrap_2d/2.png)

#### with whitening - worse results

![lab 1 - 6 - ksvm_wrap 1 - whitening.png](../6_ksvm_wrap_2d/1_whitening.png)

![lab 1 - 6 - ksvm_wrap 2 - whitening.png](../6_ksvm_wrap_2d/2_whitening.png)

## 7 - Studija slučaja: MNIST

### Za model konfiguracije [784, 10] iscrtajte i komentirajte naučene matrice težina za svaku pojedinu znamenku. Ponovite za različite iznose regularizacije

Sve konfiguracije imaju: activ=RELU, v=1, eta=0.1, early_stopping=1000 (or no early stopping)

Zanemarite crvene i plave linije na znamenakama 0, zaboravio sam jedan plt.close.

#### init=kaiming_uniform

weight decay = 1
![relu-v1-ku-wd=1](../7_mnist__digits/relu-v1-ku-wd=1.png)
weight decay = 1 (no early stopping!)
![relu-v1-ku-wd=1-force](../7_mnist__digits/relu-v1-ku-wd=1-force.png)
weight decay = 1e-1
![relu-v1-ku-wd=1e-1](../7_mnist__digits/relu-v1-ku-wd=1e-1.png)
weight decay = 1e-2
![relu-v1-ku-wd=1e-2](../7_mnist__digits/relu-v1-ku-wd=1e-2.png)
weight decay = 1e-3
![relu-v1-ku-wd=1e-3](../7_mnist__digits/relu-v1-ku-wd=1e-3.png)
weight decay = 1e-4
![relu-v1-ku-wd=1e-4](../7_mnist__digits/relu-v1-ku-wd=1e-4.png)
weight decay = 1e-4 (no early stopping!)
![relu-v1-ku-wd=1e-4-force](../7_mnist__digits/relu-v1-ku-wd=1e-4-force.png)
weight decay = 1e-5
![relu-v1-ku-wd=1e-5](../7_mnist__digits/relu-v1-ku-wd=1e-5.png)
weight decay = 0
![relu-v1-ku-wd=e0](../7_mnist__digits/relu-v1-ku-wd=e0.png)

#### init=normal

weight decay = 1
![relu-v1-normal-wd=1](../7_mnist__digits/relu-v1-normal-wd=1.png)
weight decay = 1e-3
![relu-v1-normal-wd=1e-3](../7_mnist__digits/relu-v1-normal-wd=1e-3.png)
weight decay = 1e-4
![relu-v1-normal-wd=1e-4](../7_mnist__digits/relu-v1-normal-wd=1e-4.png)
weight decay = 1e-5
![relu-v1-normal-wd=1e-5](../7_mnist__digits/relu-v1-normal-wd=1e-5.png)
weight decay = 0
![relu-v1-normal-wd=e0](../7_mnist__digits/relu-v1-normal-wd=e0.png)

#### init -- kaiming_uniform vs normal (jedno pored drugog)

weight decay = 1
normal
![relu-v1-normal-wd=1](../7_mnist__digits/relu-v1-normal-wd=1.png)
kaiming_uniform
![relu-v1-ku-wd=1](../7_mnist__digits/relu-v1-ku-wd=1.png)

Vrlo slicne. Razlike primjetim u tamnijoj/svjetlijoj pozadini kod na primjer znamenki 0 i 3.

#### Komentari

Moze se primjetiti da bez regularizacija naucene tezine ocigledno nauce sum. Uz vecu regularizaciju se sacuva ocekivani oblik znamenke.

### Naučite duboke modele s konfiguracijama [784, 10], [784, 100, 10], [784, 100, 100, 10] i [784, 100, 100, 100, 10]. Ako nemate funkcionalan GPU ne morate provoditi eksperimente s posljednje dvije konfiguracije. Nakon svake epohe učenja pohranite gubitak. Obratite pažnju na to da će dublji modeli bolje konvergirati s više iteracija s manjim korakom. Usporedite modele s obzirom na kretanje gubitka kroz epohe te pokazatelje performanse (točnost, preciznost, odziv) na skupovima za učenje i testiranje. Za najuspješniji model iscrtajte podatke koji najviše doprinose funkciji gubitka

U nastavku koristim kaiming_uniform (aka He initialization) koja se koristi i u nn.Linear jer dalje mnogo bolji inicijalni start od N(0,1). S N(0,1) sam susretao dosta nesretnih startnih pozicija koje bi (uz trenutni jednostavni postupak ucenja bez momentumama etc) ili lako zapinjale u lokalnim minimuma velikog gubitka ili dugo vremena trebale da pronadju dobro rjesenje.

Također, u svim eksperimentima koristim early stopping za izbor najboljeg modela. Prenaučene modele sam proučio u 4 podzadatku usporedbom performansi izgleda gubitka prilokom korištenja odnosno nekorištenja ranog zaustavljanja.

U nastavku su izabrani grafovi kretanja gubitka kroz epohe za različite arhitekture i različite jačine regularizacije.

#### [784, 10]

relu-normal-wd=0
![relu-normal-wd=0](../7_mnist_h=2/loss_relu-v1-normal-wd=0.png)
***
relu-normal-wd=1e-3
![relu-normal-wd=1e-3](../7_mnist_h=2/loss_relu-v1-normal-wd=1e-3.png)

#### [784, 100, 10]

relu_i=kaiming_uniform_h=3_epochs=10000_es=10000_eta=0.1_wd=0.0
![loss2_act=relu_i=kaiming_uniform_h=3_epochs=10000_es=10000_eta=0.1_wd=0.0_s=72](../7_mnist_h=3/loss2_act=relu_i=kaiming_uniform_h=3_epochs=10000_es=10000_eta=0.1_wd=0.0_s=72.png)
***
relu_i=kaiming_uniform_h=3_epochs=10000_es=10000_eta=0.1_wd=0.0001
![loss2_act=relu_i=kaiming_uniform_h=3_epochs=10000_es=10000_eta=0.1_wd=0.0001_s=360](../7_mnist_h=3/loss2_act=relu_i=kaiming_uniform_h=3_epochs=10000_es=10000_eta=0.1_wd=0.0001_s=360.png)
***
relu_i=kaiming_uniform_h=3_epochs=10000_es=10000_eta=0.1_wd=0.001
![loss2_act=relu_i=kaiming_uniform_h=3_epochs=10000_es=10000_eta=0.1_wd=0.001_s=72](../7_mnist_h=3/loss2_act=relu_i=kaiming_uniform_h=3_epochs=10000_es=10000_eta=0.1_wd=0.001_s=72.png)
***
relu_i=kaiming_uniform_h=3_epochs=100000_es=1000_eta=0.1_wd=0.001
![loss2_act=relu_i=kaiming_uniform_h=3_epochs=100000_es=1000_eta=0.1_wd=0.001_s=360](../7_mnist_h=3/loss2_act=relu_i=kaiming_uniform_h=3_epochs=100000_es=1000_eta=0.1_wd=0.001_s=360.png)
***
relu_i=kaiming_uniform_h=3_epochs=20000_es=20000_eta=0.1_wd=0.01
![loss2_act=relu_i=kaiming_uniform_h=3_epochs=20000_es=20000_eta=0.1_wd=0.01_s=72](../7_mnist_h=3/loss2_act=relu_i=kaiming_uniform_h=3_epochs=20000_es=20000_eta=0.1_wd=0.01_s=72.png)
***
relu_i=kaiming_uniform_h=3_epochs=20000_es=20000_eta=0.1_wd=0.01
![loss2_act=relu_i=kaiming_uniform_h=3_epochs=20000_es=20000_eta=0.1_wd=0.01_s=360](../7_mnist_h=3/loss2_act=relu_i=kaiming_uniform_h=3_epochs=20000_es=20000_eta=0.1_wd=0.01_s=360.png)

#### [784, 100, 100, 10]


relu_i=kaiming_uniform_h=4_epochs=100000_es=1000_eta=0.1_wd=0.0_s=360
![relu_i=kaiming_uniform_h=4_epochs=100000_es=1000_eta=0.1_wd=0.0_s=360](../7_mnist_h=4_(and_bad_h=5)/loss2_act=relu_i=kaiming_uniform_h=4_epochs=100000_es=1000_eta=0.1_wd=0.0_s=360.png)
***
relu_i=kaiming_uniform_h=4_epochs=100000_es=1000_eta=0.1_wd=0.001_s=72
![relu_i=kaiming_uniform_h=4_epochs=100000_es=1000_eta=0.1_wd=0.001_s=72](../7_mnist_h=4_(and_bad_h=5)/loss2_act=relu_i=kaiming_uniform_h=4_epochs=100000_es=1000_eta=0.1_wd=0.001_s=72.png)
***
relu_i=kaiming_uniform_h=4_epochs=100000_es=1000_eta=0.1_wd=0.01_s=72
![relu_i=kaiming_uniform_h=4_epochs=100000_es=1000_eta=0.1_wd=0.01_s=72](../7_mnist_h=4_(and_bad_h=5)/loss2_act=relu_i=kaiming_uniform_h=4_epochs=100000_es=1000_eta=0.1_wd=0.01_s=72.png)

#### [784, 100, 100, 100, 10]

Ovako izgleda funkcija gubitka kada je stopa učenja prevelika

relu_i=kaiming_uniform_h=5_epochs=100000_es=1000_eta=0.1_wd=0.0_s=192
![relu_i=kaiming_uniform_h=5_epochs=100000_es=1000_eta=0.1_wd=0.0_s=192](../7_mnist_h=4_(and_bad_h=5)/loss2_act=relu_i=kaiming_uniform_h=5_epochs=100000_es=1000_eta=0.1_wd=0.0_s=192.png)
***
relu_i=kaiming_uniform_h=5_epochs=100000_es=1000_eta=0.1_wd=0.0001_s=192
![relu_i=kaiming_uniform_h=5_epochs=100000_es=1000_eta=0.1_wd=0.0001_s=192](../7_mnist_h=4_(and_bad_h=5)/loss2_act=relu_i=kaiming_uniform_h=5_epochs=100000_es=1000_eta=0.1_wd=0.0001_s=192.png)
***
relu_i=kaiming_uniform_h=5_epochs=100000_es=1000_eta=0.1_wd=0.01_s=360
![relu_i=kaiming_uniform_h=5_epochs=100000_es=1000_eta=0.1_wd=0.01_s=360](../7_mnist_h=4_(and_bad_h=5)/loss2_act=relu_i=kaiming_uniform_h=5_epochs=100000_es=1000_eta=0.1_wd=0.01_s=360.png)
***
relu_i=kaiming_uniform_h=5_epochs=100000_es=1000_eta=0.1_wd=0.01_s=72
![relu_i=kaiming_uniform_h=5_epochs=100000_es=1000_eta=0.1_wd=0.01_s=72](../7_mnist_h=4_(and_bad_h=5)/loss2_act=relu_i=kaiming_uniform_h=5_epochs=100000_es=1000_eta=0.1_wd=0.01_s=72.png)
***
relu_i=kaiming_uniform_h=5_epochs=100000_es=1000_eta=0.1_wd=0.001_s=192
![relu_i=kaiming_uniform_h=5_epochs=100000_es=1000_eta=0.1_wd=0.001_s=192](../7_mnist_h=4_(and_bad_h=5)/loss2_act=relu_i=kaiming_uniform_h=5_epochs=100000_es=1000_eta=0.1_wd=0.001_s=192.png)
***
relu_i=kaiming_uniform_h=5_epochs=100000_es=1000_eta=0.1_wd=0.001_s=360
![relu_i=kaiming_uniform_h=5_epochs=100000_es=1000_eta=0.1_wd=0.001_s=360](../7_mnist_h=4_(and_bad_h=5)/loss2_act=relu_i=kaiming_uniform_h=5_epochs=100000_es=1000_eta=0.1_wd=0.001_s=360.png)

Smanjenjem stope učenja i povećanjem broja maksimalnih epoha sam dobio sljedeće rezultate:

relu_i=kaiming_uniform_h=5_epochs=200000_es=1000_eta=0.01_wd=0.0
![relu_i=kaiming_uniform_h=5_epochs=200000_es=1000_eta=0.01_wd=0.0](../7_mnist_h=5/loss2_act=relu_i=kaiming_uniform_h=5_epochs=200000_es=1000_eta=0.01_wd=0.0_s=192.png)
***relu_i=kaiming_uniform_h=5_epochs=200000_es=1000_eta=0.01_wd=0.0001
![relu_i=kaiming_uniform_h=5_epochs=200000_es=1000_eta=0.01_wd=0.0001](../7_mnist_h=5/loss2_act=relu_i=kaiming_uniform_h=5_epochs=200000_es=1000_eta=0.01_wd=0.0001_s=72.png)
***relu_i=kaiming_uniform_h=5_epochs=200000_es=1000_eta=0.01_wd=0.001
![relu_i=kaiming_uniform_h=5_epochs=200000_es=1000_eta=0.01_wd=0.001](../7_mnist_h=5/loss2_act=relu_i=kaiming_uniform_h=5_epochs=200000_es=1000_eta=0.01_wd=0.001_s=72.png)
***relu_i=kaiming_uniform_h=5_epochs=300000_es=2000_eta=0.03_wd=0.01
![relu_i=kaiming_uniform_h=5_epochs=300000_es=2000_eta=0.03_wd=0.01](../7_mnist_h=5/loss2_act=relu_i=kaiming_uniform_h=5_epochs=300000_es=2000_eta=0.03_wd=0.01_s=72.png)

#### Usporedba modela s obzirom na funkcije gubitka

Na MNISTU, gubitak kroz epohe pada na skupu za učenje gotovo cijelo vrijeme kod svih modela (uz razumnu stopu učenja za koju ne dolazi do divergiranja i za koju model ne uči presporo). Ponašanje pogreške na skupu za validaciju ovisi o jačini regularizacije. Za jaču regularizaciju funkcija pogreške na skupu za treniranje i validaciju je bliža nego kada se regularizacija ne koristi.

#### Pokazatelji performanse (točnost, preciznost, odziv) na skupovima za učenje i testiranje

Arhitekture sam pokrenuo po 3 puta za svaku konfiguraciju (uz različite seedove / incijalne vrijednosti težina) da bih dobio mean i std (osim [784,10] koju sam pokrenuo samo jednom).

Pokazateljii performanse su prikazani za svaku arhitekturu odvojeno:

<!-- ![h=2_perf_mean_2](../7_mnist__h=2_perf_mean_2.png) -->
![h=2_perf_mean](../7_mnist__h=2_perf_mean.png)
![h=3_perf_mean](../7_mnist__h=3_perf_mean.png)
![h=3_perf_std](../7_mnist__h=3_perf_std.png)
![h=4_perf_mean](../7_mnist__h=4_perf_mean.png)
![h=4_perf_std](../7_mnist__h=4_perf_std.png)
![h=5_perf_mean](../7_mnist__h=5_perf_mean.png)
![h=5_perf_std](../7_mnist__h=5_perf_std.png)

U sljedećoj tablici i pripadnog grafu su vizualizirane test performance za sve konfiguracije. Najbolja konfiguracija je istaknuta.

![test accs](2021-02-10-00-31-34.png)
![test_comparison](../7_mnist__test_comparison.png)

Arhitektura [784,100,100,10] s wd=1e-3 je dala najbolju generalizacijsku performansu. Arhitekture s jednim skrivenim slojem više i manje od [784,100,100,10] daju također vrlo dobre rezultate.

Najuspješniji model: `python.exe -m demo.mlp.mnist_shootout --init kaiming_uniform --log_interval 1 --log_images_interval 400000 --v 1 --npl 784 100 100 10 --epochs 100000 --es 1000 --wd 1e-3 --eta 0.1 --seed 360`

Loss i acc za najuspješniji model:
![accuracy2](../7_mnist__best_model/accuracy2.png)
![loss2](../7_mnist__best_model/loss2.png)


#### Najteži podatci

Najteži po podskupovima:
![train_0_i=35480_loss=3.413605_true=9](../7_mnist__best_model/train_0_i=35480_loss=3.413605_true=9.png)
![valid_0_i=9915_loss=13.993727_true=4](../7_mnist__best_model/valid_0_i=9915_loss=13.993727_true=4.png)
![test_0_i=1247_loss=9.334496_true=9](../7_mnist__best_model/test_0_i=1247_loss=9.334496_true=9.png)

4x4 gridovi najtežih podataka po podskupu:

Train:
![train_grid](../7_mnist__best_model/train_grid.png)

Valid:
![valid_grid](../7_mnist__best_model/valid_grid.png)

Test:
![test_grid](../7_mnist__best_model/test_grid.png)

### Proučite utjecaj regularizacije na performansu dubokih modela na skupovima za učenje i testiranje

Na grafovima iz prethodnog podzadatka vidimo da razumna količina regulariacje pogoduje generalizaciji. Za naš skup podataka i ostale parametre su 1e-2 i 1e-3 među pogodnim vrijednostima za regularizaciju. Pretjerana regularizacija je onemogućila učenje i naškodila performansi modela (wd=0.1 i wd=1). Jasno je da je regularizacija hiperparametar.

#### Standardizacija ulaza kao oblik regularizacije

Svi eksperimenti su radili sa standardiziaranom verzijom MNISTa:
```py
class PTUtil:
      # ...
      @staticmethod
      def standardize_inplace(x_train, x_rest=[]):
            train_mean = x_train.mean()
            train_std = x_train.std()

            x_train.sub_(train_mean).div_(train_std)
            for x in x_rest:
                  x.sub_(train_mean).div_(train_std)

            return train_mean, train_std
      # ...

if params["whiten input"]:
      train_mean, train_std = PTUtil.standardize_inplace(x_train, [x_valid, x_test])
```

Nisam testirao koliki je utjecaj ovoga u poboljšanju generalizacijske sposbnosti mreže, nego sam podatke standardizirao sljedeći upute dobre prakse. Standardizacija bi trebala poboljšati uvijete za gradijenti spust i olakšati pronalazak boljeg minimuma, time poboljšavajući generalizacijske sposobnosti (stoga se naziva regularizacijom).

### Slučajno izdvojite 1/5 podataka iz skupa za učenje u skup za validaciju. Tijekom treniranja evaluirajte validacijsku performansu nakon završetka petlje po grupama podataka te na kraju vratite model s najboljom validacijskom performansom (engl. early stopping). Procijenite postignuti utjecaj na konačnu vrijednost funkcije cilja i generalizacijsku performansu

Implementirano rano zaustavljanje ce se prekinuti treniranje ako nakon X epoha validation_loss ne padne za barem Y (koristio sam X=1000 ili 2000 i Y=1e-7).

Pokrenuo sam najuspješniju arhitekturu uz wd=0 i wd=1e-3 i ostavio da izvrte 300k epoha do kraja, u nadi da će se prenaučiti:
`python.exe -m demo.mlp.mnist_shootout --init kaiming_uniform --log_interval 100 --log_images_interval 400000 --v 1 --npl 784 100 100 10 --epochs 300000 --es 1000000000 --wd 1e-3 --eta 0.1 --seed 360 --prefix force_`
`python.exe -m demo.mlp.mnist_shootout --init kaiming_uniform --log_interval 100 --log_images_interval 400000 --v 1 --npl 784 100 100 10 --epochs 300000 --es 1000000000 --wd 0 --eta 0.1 --seed 360 --prefix force_`

Dodatno:
`python.exe -m demo.mlp.mnist_shootout --init kaiming_uniform --log_interval 100 --log_images_interval 400000 --v 1 --npl 784 100 100 10 --epochs 300000 --es 1000000000 --wd 1e-3 --eta 0.1 --seed 72 --prefix force_`

Rezultati early stopping modela u usporedbi s modelom koji je odradio svih 300k epoha:

**wd=0.0**: Ovaj se model moze prenauciti i to se dogodilo sudeci po grafu funckije gubitka odnosno ponasanju validacijske funkcije gubitka (valid. funkcija gubitka raste dok na skupu za treniranje pada). U evaluaciji generalizacijskih sposobnosti na skupu za testiranje, to se odrazilo u duplo vecoj vrijednosti funkcije pogreske na prenaucenom modelu. Valja primjetiti da su, unatoc vecoj vrijednosti funkcije gubitka, u ovom eksperimentu tocnost i f1 mjera bolje u prenaucenom modelu. Ovo nisam ocekivao za prenauceni model.
<!-- i mislim da se moze opravdati time da je samo jedan eksperiment pokrenut, bez ikakve mjere drugog momenta i da bi temeljitije testiranje dalo bolje temelje za zakljucivanje. Takodjer, ovo moze biti uzrokovano implementiranom verzijom ranog zaustavljanja koja u obzir uzima samo loss na valid setu, ne i accuracy. -->
Update: Ponovio sam eksperiment s drugim seedom, ali pronašao isti trend: funkcija gubitka na prenaučenom modelu je duplo gora ali je točnost viša.

Log loss curve:
![Log loss curve](../7_mnist_force/force_opt=sgd_act=relu_i=kaiming_uniform_h=4_mb=50000_epochs=300000_es=1000000000_eta=0.1_wd=0.0_s=360/loss2.png)

*Selected performance metrics:*
| metric | end/forced    | best/early_stopping |
|--------|------|-------|
| epochs | 300000 | **3100** |
| test_acc     | **0.9781** | 0.9747 |
| test_f1      | **0.9779** | 0.9744 |
| valid_acc    | **0.9768** | 0.9766 |
| train_acc    | **1.0** | 0.9962 |
| test_loss | 0.167346 | **0.081083** |
| train_loss| **0.000013** | 0.021577 |

All metrics:
*Early stopping:*

```js
{'best_epoch': 3100,
 'best_test_accuracy': 0.9747,
 'best_test_f1': 0.9744698355524429,
 'best_test_loss': 0.08108333142707712,
 'best_test_precision': 0.9745165826594324,
 'best_test_recall': 0.9744230929301125,
 'best_train_accuracy': 0.99624,
 'best_train_f1': 0.9962650926270269,
 'best_train_loss': 0.021577165637300905,
 'best_train_precision': 0.9962684886907056,
 'best_train_recall': 0.9962616965865008,
 'best_valid_accuracy': 0.9766,
 'best_valid_f1': 0.976384090824407,
 'best_valid_loss': 0.08056664346910276,
 'best_valid_precision': 0.9765197519388122,
 'best_valid_recall': 0.9762484673976811}
 ```

*Forced to do all epochs:

```js
{'end_test_accuracy': 0.9781,
 'end_test_f1': 0.9779084999998212,
 'end_test_loss': 0.1673462301429217,
 'end_test_precision': 0.9779277762105405,
 'end_test_recall': 0.9778892245490048,
 'end_train_accuracy': 1.0,
 'end_train_f1': 1.0,
 'end_train_loss': 1.3195074273106684e-05,
 'end_train_precision': 1.0,
 'end_train_recall': 1.0,
 'end_valid_accuracy': 0.9768,
 'end_valid_f1': 0.97659190576326,
 'end_valid_loss': 0.15827165365456036,
 'end_valid_precision': 0.9767867261015049,
 'end_valid_recall': 0.9763971631234412}
```

**wd=0.001**: Uz ovu količinu regularizacije, model se nije uspio prenaučiti. Perfomanse na skupu za testiranje su slične bez i s ranim zaustavljanjem.
*Log loss curve*
![Log loss curve](../7_mnist_force/force_opt=sgd_act=relu_i=kaiming_uniform_h=4_mb=50000_epochs=300000_es=1000000000_eta=0.1_wd=0.001_s=360/loss2.png)

*Selected performance metrics:*
| metric | end/forced    | best/early_stopping |
|--------|------|-------|
| epochs | 300000 | **73705** |
| test_acc     | 0.9812 | **0.982** |
| test_f1      | 0.9811 | **0.9819** |
| vald_acc    | 0.9820 | 0.9820 |
| train_acc    | **0.9984** | 0.9983 |
| test_loss | **0.136502** | 0.138197 |
| train_loss| **0.095400** | 0.097330 |

All metrics:
*Early stopping:*

```js
{'best_epoch': 73705,
 'best_test_accuracy': 0.982,
 'best_test_f1': 0.9818980785122735,
 'best_test_loss': 0.13819733055148867,
 'best_test_precision': 0.9819536120021628,
 'best_test_recall': 0.9818425513033207,
 'best_train_accuracy': 0.99828,
 'best_train_f1': 0.9983037329038189,
 'best_train_loss': 0.09733061525997733,
 'best_train_precision': 0.998306631234309,
 'best_train_recall': 0.9983008345901577,
 'best_valid_accuracy': 0.982,
 'best_valid_f1': 0.9818685837335382,
 'best_valid_loss': 0.1393339890208909,
 'best_valid_precision': 0.9820158063010412,
 'best_valid_recall': 0.9817214053022607}
 ```

*Forced to do all epochs:*

```js
{'end_test_accuracy': 0.9812,
 'end_test_f1': 0.9810631342508451,
 'end_test_loss': 0.13650270435846934,
 'end_test_precision': 0.9811701762616923,
 'end_test_recall': 0.9809561155932192,
 'end_train_accuracy': 0.99844,
 'end_train_f1': 0.9984710406150795,
 'end_train_loss': 0.09540081636818312,
 'end_train_precision': 0.9984775505084004,
 'end_train_recall': 0.9984645308066444,
 'end_valid_accuracy': 0.982,
 'end_valid_f1': 0.9818671373890815,
 'end_valid_loss': 0.13699347088921496,
 'end_valid_precision': 0.9819635198923979,
 'end_valid_recall': 0.9817707738043401}
```

### Implementirajte stohastički gradijentni spust odnosno postupak učenja po minigrupama. Prije svake epohe izmiješajte podatke, zatim ih podijelite u n grupa (engl. mini-batch) i onda provedite korak učenja za svaku grupu posebno. Pripazite na to da gubitak karakterizirate tako da ne ovisi o veličini grupe jer je tako lakše interpretirati iznos gubitka te validirati korak učenja. Vaš kod pohranite u metodi train_mb. Procijenite utjecaj na kvalitetu konvergencije i postignutu performansu za najuspješniju konfiguraciju iz prethodnog zadatka. Napomena: u svrhu razumijevanja postupka učenja po mini-grupama, u ovoj vježbi nije dozvoljeno korištenje razreda torch.utils.data.DataLoader

Sav kod sam pohranio u metodu train (ne u train_mb) koja je parametrizirana parametrima params koji određuju kako će se odviti učenje.

Pokrenuo sam:
`python.exe -m demo.mlp.mnist_shootout --init kaiming_uniform --log_interval 1 --log_images_interval 400000 --v 1 --npl 784 100 100 10 --epochs 100000 --es 2000 --wd 1e-3 --eta 0.1 --seed 360 --prefix minibatch_ --bs 128`
Log loss for batch_size=128
![mb=128](../7_mnist_batch_size/logloss_minibatch_opt=sgd_act=relu_i=kaiming_uniform_h=4_mb=128_epochs=100000_es=200_eta=0.1_wd=0.001_s=360.png)
<!-- Nije potpun ovaj loss plot jer pokazuje run za early_stopping=200, fali 1800 iteracija do 2000, zagubio sam ih :-) Ali early stopping bi na istom mjestu stao i sa 2000-->
***
`python.exe -m demo.mlp.mnist_shootout --init kaiming_uniform --log_interval 1 --log_images_interval 400000 --v 1 --npl 784 100 100 10 --epochs 100000 --es 2000 --wd 1e-3 --eta 0.1 --seed 360 --prefix minibatch_ --bs 1024`
Log loss for batch_size=1024
![logloss_minibatch_opt=sgd_act=relu_i=kaiming_uniform_h=4_mb=1024_epochs=100000_es=2000_eta=0.1_wd=0.001_s=360](../7_mnist_batch_size/logloss_minibatch_opt=sgd_act=relu_i=kaiming_uniform_h=4_mb=1024_epochs=100000_es=2000_eta=0.1_wd=0.001_s=360.png)
***
`python.exe -m demo.mlp.mnist_shootout --init kaiming_uniform --log_interval 1 --log_images_interval 400000 --v 1 --npl 784 100 100 10 --epochs 100000 --es 2000 --wd 1e-3 --eta 0.1 --seed 360 --prefix minibatch_ --bs 4096`
Log loss for batch_size=4096
![logloss_minibatch_opt=sgd_act=relu_i=kaiming_uniform_h=4_mb=4096_epochs=100000_es=2000_eta=0.1_wd=0.001_s=360](../7_mnist_batch_size/logloss_minibatch_opt=sgd_act=relu_i=kaiming_uniform_h=4_mb=4096_epochs=100000_es=2000_eta=0.1_wd=0.001_s=360.png)

Nisam mjerio drugi moment, vec samo pokrenuo po jedan eksperiment za svaki batch size (128,1024,4096,60000) uz early stopping nakon 2000 epoha bez poboljsanja.

<!-- Metrics comparison for different values of batch size -->
![metrics_comparison](../7_mnist_batch_size/metrics_comparison.png)
![metrics_comparison](../7_mnist_batch_size/convergence_speed.png)

Gotovo jednako dobra generalizacijska sposobnost, ali u puno manje koraka. Intuitivno, manji batch size (krajnost je batch_size=1, tj. stohastic gradient descent) uvodi noise koji pogoduje pronalasku boljeg minimuma i napustanju loseg lokalnog minimuma, poboljsavajuci izglede pronalaka boljeg minimuma i pretrazivanja veceg dijela prostora tezina (weight space).

Noise i stohasticnost s manjim batch sizeom se moze vidjeti na grafu funkcija gubitaka za batch_size=128 koja nije glatka kao u slucaju ucenja na cijelom batchu. Za batch_size=1024 i batch_size=4096 se ne primjeti ovo ponašanje na prikazanim grafovima, što ne implicira da ne postoji.

### Promijenite optimizator u torch.optim.Adam s fiksnim korakom učenja 1e-4. procijenite utjecaj te promjene na kvalitetu konvergencije i postignutu performansu

`python.exe -m demo.mlp.mnist_shootout --init kaiming_uniform --log_interval 1 --log_images_interval 400000 --v 1 --npl 784 100 100 10 --epochs 150000 --es 2000 --wd 1e-3 --eta 0.001 --seed 360 --optimizer adam`
<!-- `python.exe -m demo.mlp.mnist_shootout --init kaiming_uniform --log_interval 1 --log_images_interval 400000 --v 1 --npl 784 100 100 10 --epochs 150000 --es 2000 --wd 1e-3 --eta 0.1 --seed 360 --optimizer adam` -->
![logloss_opt=adam_act=relu_i=kaiming_uniform_h=4_mb=50000_epochs=150000_es=2000_eta=0.001_wd=0.001_s=360](../7_mnist_optimizers/logloss_opt=adam_act=relu_i=kaiming_uniform_h=4_mb=50000_epochs=150000_es=2000_eta=0.001_wd=0.001_s=360.png)

convergence_speed
![convergence_speed](../7_mnist_optimizers/convergence_speed.png)

metrics_comparison
![metrics_comparison](../7_mnist_optimizers/metrics_comparison.png)

U pokrenutom eksperimentu, ADAM optimizator je postigao neznatno losije rezultate od modela treniranog SGDom, ali je mnogo brze konvergirao (odnosno zaustavio se - early stopping). Na grafu log lossa se vidi da ucenje nije bilo sasvim glatko.

### Isprobajte ADAM s varijabilnim korakom učenja. U izvedbi se pomognite funkcijom torch.optim.lr_scheduler.ExponentialLR, koju valja pozvati nakon svake epohe kao što je preporučeno u dokumentaciji). Neka početni korak učenja bude isti kao i ranije, a ostale parametre postavite na gamma=1-1e-4

`python.exe -m demo.mlp.mnist_shootout --init kaiming_uniform --log_interval 1 --log_images_interval 400000 --v 1 --npl 784 100 100 10 --epochs 150000 --es 2000 --wd 1e-3 --eta 0.001 --seed 360 --optimizer adam+ls`
<!-- `python.exe -m demo.mlp.mnist_shootout --init kaiming_uniform --log_interval 1 --log_images_interval 400000 --v 1 --npl 784 100 100 10 --epochs 150000 --es 2000 --wd 1e-3 --eta 0.1 --seed 360 --optimizer adam+ls` -->
![logloss_opt=adam+ls_act=relu_i=kaiming_uniform_h=4_mb=50000_epochs=150000_es=2000_eta=0.001_wd=0.001_s=360](../7_mnist_optimizers/logloss_opt=adam+ls_act=relu_i=kaiming_uniform_h=4_mb=50000_epochs=150000_es=2000_eta=0.001_wd=0.001_s=360.png)

Konvergencija funckije gubitka je glatka. ADAM+LS je postigao neznatno losije rezultate od SGDa i ADAMa, ali je imao znacajno brze rano zaustavljanje od modela treniranog SGDom.

### Izračunajte i interpretirajte gubitak slučajno incijaliziranog modela (dakle, modela koji nije vidio podatke za učenje)

Gubitak slucajno inicijaliziranog modela:
normal, weight_decay=0
2,3,4,5
29.481732554340915
32.915821019002294
33.692973614216776
33.23203412270679

kaiming_uniform, weight_decay=0
2,3,4,5
2.3164354374945164
2.3779982983255388
2.296685476255417
2.300553603901863

Ako imamo model koji daje jednake vjerojojatnosti za sve razrede i gubitak bez regularizacije, tada bi gubitak iznosio -1\*ln(0.1) = 2.303. Za kaiming_uniform inicijaliziciju, ishod je vrlo slican, dok za normalnu inicijalizaciju nije. To bi znacilo da potonja inicijalizicja daje distribuciju oznaka koja je znacajno losija one slucajno ocekivane i vjerojatno ima mnogo slucjeva gdje je predvidjena vjerojatnost bliska nuli (npr -ln(1e-20)=46).

### Naučite linearni i jezgreni SVM uz pomoć modula sklearn.svm. Koristite podrazumijevano one vs one proširenje SVM-a za klasi ciranje podataka u više razreda. Pri eksperimentiranju budite strpljivi jer bi učenje i evaluacija mogli trajati više od pola sata. Usporedite dobivenu performansu s performansom dubokih modela

<!--
python.exe -m demo.svm.ksvm_wrap_mnist --cs 1 --gammas auto & python.exe -m demo.svm.ksvm_wrap_mnist --cs 1 --gammas 3 & python.exe -m demo.svm.ksvm_wrap_mnist --cs 1 --gammas 1 & python.exe -m demo.svm.ksvm_wrap_mnist --cs 1 --gammas 0.1 & python.exe -m demo.svm.ksvm_wrap_mnist --cs 5 --gammas auto & python.exe -m demo.svm.ksvm_wrap_mnist --cs 5 --gammas 3 & python.exe -m demo.svm.ksvm_wrap_mnist --cs 5 --gammas 1 & python.exe -m demo.svm.ksvm_wrap_mnist --cs 5 --gammas 0.1 & python.exe -m demo.svm.ksvm_wrap_mnist --cs 10 --gammas auto & python.exe -m demo.svm.ksvm_wrap_mnist --cs 10 --gammas 3 & python.exe -m demo.svm.ksvm_wrap_mnist --cs 10 --gammas 1 & python.exe -m demo.svm.ksvm_wrap_mnist --cs 10 --gammas 0.1 &

python -m demo.svm.ksvm_wrap_mnist --cs 100 --gammas auto
python -m demo.svm.ksvm_wrap_mnist --cs 100 --gammas 3
python -m demo.svm.ksvm_wrap_mnist --cs 100 --gammas 1
python -m demo.svm.ksvm_wrap_mnist --cs 100 --gammas 0.1
python -m demo.svm.ksvm_wrap_mnist --cs 1000 --gammas auto
python -m demo.svm.ksvm_wrap_mnist --cs 1000 --gammas 3
python -m demo.svm.ksvm_wrap_mnist --cs 1000 --gammas 1
python -m demo.svm.ksvm_wrap_mnist --cs 1000 --gammas 0.1
-->

RBF SVM je uzasno sporiji, sto je ocekivano s obzirom na kompleksnost SVMa s brojem podatka. SVM sam trenirao bez validacijskog seta, dakle na svih 60k slika. Prostor hiperparametara je doduse bilo jednostavnije pretraziti nego za konfigurabilne duboke modele. Pokrenuo sam samo nekoliko konfiguracija koje su se mogle izvrtiti u izvjesnom vremenu. Rezultati su u nastavku:

| kernel | c    | gamma | test_accuracy | test_precision | test_recall | train_accuracy | test_loss | train_loss |
|--------|------|-------|---------------|----------------|-------------|----------------|-----------|------------|
| rbf    | 0.1  | auto  | 92.10%        | 92.00%         | 91.99%      | 91.64%         | 0.254681  | 0.272119   |
| rbf    | 1    | auto  | 94.62%        | 94.56%         | 94.54%      | 94.46%         | 0.180943  | 0.185335   |
| rbf    | 5    | auto  | 95.80%        | 95.76%         | 95.74%      | 96.34%         | 0.140595  | 0.129842   |
| rbf    | 10   | auto  | 96.29%        | 96.26%         | 96.24%      | 97.08%         | 0.123166  | 0.103742   |
| rbf    | 100  | auto  | 97.72%        | 97.70%         | 97.69%      | 99.49%         | 0.081970  |            |
| rbf    | 1000 | auto  | **97.74%**        | **97.72%**        | **97.72%**      | **100.00%**        | **0.081040**  |            |
| linear |    1 | auto  | 94.37%        | 94.29%         | 94.29%      |         97.28% |  0.203150 |   0.136047 |

<!--
csv:
kernel,c,gamma,test_accuracy,test_precision,test_recall,train_accuracy,test_loss,train_loss
rbf,0.1,auto,0.921,0.9199979513803747,0.9199157993205505,0.9164166666666667,0.2546809417862533,0.2721189933371896
rbf,1,auto,0.9462,0.945551309191861,0.9453768618801,0.94455,0.180943119545632,0.18533505756435
linear,1,auto,0.9437,0.9429127857957227,0.9429138580734128,0.9727666666666667,0.20314965959270936,0.1360465293682513
rbf,5,auto,0.958,0.957599222991113,0.957373220009433,0.963383333333333,0.140595415425771,0.129842191952165
rbf,10,auto,0.9629,0.962615667558653,0.962417317677036,0.970766666666667,0.123165648147696,0.103741938398826
rbf,100.0,auto,0.9772,0.9769910924580769,0.9769041481394115,0.9948666666666667,0.08197
rbf,1000.0,auto,0.9774,0.977227343680698,0.9771746420259028,0.9999666666666667,0.08104
 -->

SVM postize nešto lošiju točnost na skupu za testiranje od najbolje duboke konfguracije: 97.74% vs 98.17%

## 8 - Bonus zadatci

### BatchNorm

Slijedio sam pseudokod i primjer opisan ovdje: <https://gluon.mxnet.io/chapter04_convolutional-neural-networks/cnn-batch-norm-scratch.html>

Formule iz "Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift" (Sergey Ioffe, Christian Szegedy)
$$\mu_B \leftarrow \frac{1}{m}\sum_{i = 1}^{m}x_i$$
$$\sigma_B^2 \leftarrow \frac{1}{m} \sum_{i=1}^{m}(x_i - \mu_B)^2$$
$$\hat{x_i} \leftarrow \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}$$
$$y_i \leftarrow \gamma \hat{x_i} + \beta \equiv \text{BN}_{\gamma,\beta}(x_i)$$

<!-- Learnable parameters $\gamma$ i $\beta$ (afinu transformaciju normalizirane vrijednosti) nisam implementirao. Ma ne implementirat cu, imam autograd, nes ti -->
Parametar $\epsilon$ sam postavio na 1e-5.

`python.exe -m demo.mlp.mnist_shootout --init kaiming_uniform --log_interval 1 --log_images_interval 400000 --v 1 --npl 784 100 100 10 --epochs 150000 --es 10000 --wd 0 --eta 0.01 --optimizer sgd --prefix batchnorm_ --batch_norm True --batch_norm_epsilon 1e-5 --batch_norm_momentum 0.9 --seed 360`
![logloss_batchnorm_opt=sgd_act=relu_i=kaiming_uniform_h=4_mb=50000_epochs=150000_es=10000_eta=0.01_wd=0.0_s=360](../7_mnist_batchnorm/logloss_batchnorm_opt=sgd_act=relu_i=kaiming_uniform_h=4_mb=50000_epochs=150000_es=10000_eta=0.01_wd=0.0_s=360.png)

`python.exe -m demo.mlp.mnist_shootout --init kaiming_uniform --log_interval 1 --log_images_interval 400000 --v 1 --npl 784 100 100 10 --epochs 150000 --es 10000 --wd 1e-3 --eta 0.01 --optimizer sgd --prefix batchnorm_ --batch_norm True --batch_norm_epsilon 1e-5 --batch_norm_momentum 0.9 --seed 360`
![logloss_batchnorm_opt=sgd_act=relu_i=kaiming_uniform_h=4_mb=50000_epochs=150000_es=10000_eta=0.01_wd=0.001_s=360](../7_mnist_batchnorm/logloss_batchnorm_opt=sgd_act=relu_i=kaiming_uniform_h=4_mb=50000_epochs=150000_es=10000_eta=0.01_wd=0.001_s=360.png)


Usporedba perfomansi istih konfiguarcija sa i bez batchnorma (testirao sam samo [784,100,100,10] sa wd=0.0 i sa wd=0.001):

![convergence_speed](../7_mnist_batchnorm/convergence_speed.png)

![metrics_comparison](../7_mnist_batchnorm/metrics_comparison.png)

Dakle ove konfiguracije su prikazane:

```js
act=relu_i=kaiming_uniform_h=4_epochs=100000_es=1000_eta=0.1_wd=0.0
batchnorm_opt=sgd_act=relu_i=kaiming_uniform_h=4_mb=50000_epochs=150000_es=10000_eta=0.01_wd=0.0
act=relu_i=kaiming_uniform_h=4_epochs=100000_es=1000_eta=0.1_wd=0.001
batchnorm_opt=sgd_act=relu_i=kaiming_uniform_h=4_mb=50000_epochs=150000_es=10000_eta=0.01_wd=0.001
```

Nisam siguran je li bi ocekivao bolje rezultate kod batchnorma na ova dva pokrenuta primjera, ali iz rezultata je vidljivo da je test performansa losija.

Trebao sam isprobati vise hiperparametara i jos druge modele za zakljucivati vise. Batchnorm implementaciju nisam pomno testirao, mozda ima gdjegod koja prikrivena buba. (TODO testirati temeljitije ako imam vremena)

<!-- ### Anizotropna regularizacija

Sto je to? -->
