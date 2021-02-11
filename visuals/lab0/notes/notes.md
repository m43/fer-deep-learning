# Lab 0 notes (ušporko)

Excel sheet with some results:
<https://docs.google.com/spreadsheets/d/1fv7UyF2UmWiX7Lin4GtoRyjfZfZsXrGy9lFEeq9bXfQ/edit#gid=0>

## 2.1 Provjerite poklapaju li se vrijednosti analitičkih gradijenata s njihovim numeričkim aproksimacijama (engl. gradient checking)

I think they do. They only differ if the analytical gradients are too small, but otherwise are approximately equal.
The case when the analytical gradients are very small (for params={"epochs": 100000, "eta": 1, "lambda": 0}):
---analytical gradients: [ 1.04877705e-13  1.16686283e-15 -4.63711292e-13]
---numerical gradients: [ 7.97972799e-12  6.24500451e-12 -3.46944695e-13]
---Gradcheck grad difference (should be below 1e-6): 0.9468835403689254
This is how i do grad_check:
dw = 1e-5 (for each w_i separately, the bias weight included)
numerical_grads = (loss(w+dw) - loss(w-dw)) / (2*dw) --Dx1
diff = |grads-numerical_grads| / ( |grads| + |numerical_grads| )  # Andrew Ng, <http://cs230.stanford.edu/files/C2M1_old.pdf>
"Good" if (diff <= cca 1e-8) else "Bad"

## 2.2 Dodajte regularizacijski gubitak u obliku L2 norme težina w pomnožene hiperparametrom param_lambda (parametre b  najčešće nema smisla regularizirati). Ne zaboravite izraziti utjecaj te promjene na gradijente

 Larger loss, but weights closer to zero:
before:
 'accuracy': 0.965,
 'loss': 0.0621127106986851,
 'w_mean': -0.9367516145531858,
 'w_std': 2.8487826797504705
after (lambda=0.01)
 'accuracy': 0.96,
 'loss': 0.11441632779277375,
 'w_mean': -0.23347257412606948,
 'w_std': 1.1868119294413713

## 2.3 Eksperimentirajte s različitim vrijednostima hiper-parametara epochs, eta i lambda. Pronađite kombinacije  hiper-parametara za koje vaš program ne uspijeva pronaći zadovoljavajuće rješenje i objasnite što se događa

name: ex01
params:
    {"epochs": 1000, "eta": 100, "lambda": 0}
results:
{'accuracy': 0.965,
 'loss': 4.336190654272568,
 'precision': 0.9603960396039604,
 'recall': 0.97,
 'w': array([-140.15251288,   82.91952849]),
 'w_mean': -28.616492194262364,
 'w_std': 111.53602068559123}
comment: "overshooting" the minimum, though not very bad as it does not diverge

name: ex02
params:
    {"epochs":10, "eta":0.1, "lambda":0}
results:
{'accuracy': 0.955,
 'loss': 0.15517174386013166,
 'precision': 1.0,
 'recall': 0.91,
 'w': array([-1.90737086,  1.05848016]),
 'w_mean': -0.4244453468708167,
 'w_std': 1.4829255107500334}
comment: not trained for enough epochs. compare to experiment below

name: ex03
params:
    {"epochs":10000, "eta":0.1, "lambda":0}
results:
{'accuracy': 0.965,
 'loss': 0.06524058382834898,
 'precision': 0.9696969696969697,
 'recall': 0.96,
 'w': array([-3.00666517,  2.09776666]),
 'w_mean': -0.4544492542413696,
 'w_std': 2.552215916333711}

name: ex04
params:
    {"epochs":10000, "eta":2, "lambda":0}
results:
{'accuracy': 0.965,
 'loss': 0.062069111366528436,
 'precision': 0.9696969696969697,
 'recall': 0.96,
 'w': array([-3.9072773 ,  1.90815627]),
 'w_mean': -0.9995605171867253,
 'w_std': 2.907716784270339}
comment: eta parameter closer to optimum value, makes the convergence faster (compared to ex03)

name: ex05
params:
    {"epochs":10000, "eta":1, "lambda":1}
results:
{'accuracy': 0.5,
 'loss': 832727537.2017386,
 'precision': nan,
 'recall': 0.0,
 'w': array([-16218.68821527, -23871.19071281]),
 'w_mean': -20044.939464037892,
 'w_std': 3826.251248771826}
comment: too much regularization

## 2.4 Smislite način za vizualiziranje plohe funkcije gubitka i napredovanje postupka optimizacije

Added animations, set params["animation"]=True. Saved to params["save_dir"].

## 2.5 Procijenite pristranost (eng. bias) i varijancu vašeg postupka većim brojem eksperimenata na podatcima za  testiranje. Podatke za testiranje dobijte uzorkovanjem iste podatkovne distribucije koja je korištena za dobivanje  skupa za učenje. Možete li zadati distribuciju podataka za koju će pristranost klasifikatora biti vrlo velika?

 TODO
Move this task to another demo.
run many experiments with same config but different datapoints from the same, true distribution. variance should
be, hmmm.. The stddev of all individual weights, then summed? And the bias as the difference from many test datapoints?
TODO google how to estimate model bias and variance
