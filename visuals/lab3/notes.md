# Lab 3 notes (ušporko)

## 1. zadatak -- Učitavanje podataka (25%)

✓
## 2. zadatak -- Implementacija baseline modela (25%)

✓

<details>
    <summary> Baseline performance -- one run</summary>

```js
TRAIN --> avg_loss=0.650956928730011	acc=0.5443641543388367
[0/100] VALID RESULTS
{'loss': tensor(0.5656), 'acc': tensor(0.5991), 'pre': tensor(0.8730), 'rec': tensor(0.2336), 'f1': tensor(0.3685), 'confmat': [[tensor(213), tensor(31)], [tensor(699), tensor(878)]]}
TRAIN --> avg_loss=0.5173309445381165	acc=0.709104061126709
[1/100] VALID RESULTS
{'loss': tensor(0.4764), 'acc': tensor(0.7419), 'pre': tensor(0.8530), 'rec': tensor(0.5855), 'f1': tensor(0.6944), 'confmat': [[tensor(534), tensor(92)], [tensor(378), tensor(817)]]}
TRAIN --> avg_loss=0.47405898571014404	acc=0.7653179168701172
[2/100] VALID RESULTS
{'loss': tensor(0.4671), 'acc': tensor(0.7419), 'pre': tensor(0.8647), 'rec': tensor(0.5746), 'f1': tensor(0.6904), 'confmat': [[tensor(524), tensor(82)], [tensor(388), tensor(827)]]}
TRAIN --> avg_loss=0.458819180727005	acc=0.7770231366157532
[3/100] VALID RESULTS
{'loss': tensor(0.4664), 'acc': tensor(0.7518), 'pre': tensor(0.8628), 'rec': tensor(0.5998), 'f1': tensor(0.7076), 'confmat': [[tensor(547), tensor(87)], [tensor(365), tensor(822)]]}
TRAIN --> avg_loss=0.4498438239097595	acc=0.7826589345932007
[4/100] VALID RESULTS
{'loss': tensor(0.4653), 'acc': tensor(0.7540), 'pre': tensor(0.8494), 'rec': tensor(0.6184), 'f1': tensor(0.7157), 'confmat': [[tensor(564), tensor(100)], [tensor(348), tensor(809)]]}
TRAIN --> avg_loss=0.4443525969982147	acc=0.7878612875938416
[5/100] VALID RESULTS
{'loss': tensor(0.4611), 'acc': tensor(0.7661), 'pre': tensor(0.8223), 'rec': tensor(0.6798), 'f1': tensor(0.7443), 'confmat': [[tensor(620), tensor(134)], [tensor(292), tensor(775)]]}
TRAIN --> avg_loss=0.43979182839393616	acc=0.7943641543388367
[6/100] VALID RESULTS
{'loss': tensor(0.4724), 'acc': tensor(0.7507), 'pre': tensor(0.8567), 'rec': tensor(0.6031), 'f1': tensor(0.7079), 'confmat': [[tensor(550), tensor(92)], [tensor(362), tensor(817)]]}
TRAIN --> avg_loss=0.4355895519256592	acc=0.7936416268348694
[7/100] VALID RESULTS
{'loss': tensor(0.4757), 'acc': tensor(0.7474), 'pre': tensor(0.8553), 'rec': tensor(0.5965), 'f1': tensor(0.7028), 'confmat': [[tensor(544), tensor(92)], [tensor(368), tensor(817)]]}
TRAIN --> avg_loss=0.43226274847984314	acc=0.7988439202308655
[8/100] VALID RESULTS
{'loss': tensor(0.4766), 'acc': tensor(0.7518), 'pre': tensor(0.8433), 'rec': tensor(0.6195), 'f1': tensor(0.7143), 'confmat': [[tensor(565), tensor(105)], [tensor(347), tensor(804)]]}
TRAIN --> avg_loss=0.4312356114387512	acc=0.8004335165023804
[9/100] VALID RESULTS
{'loss': tensor(0.4663), 'acc': tensor(0.7540), 'pre': tensor(0.8402), 'rec': tensor(0.6283), 'f1': tensor(0.7189), 'confmat': [[tensor(573), tensor(109)], [tensor(339), tensor(800)]]}
TRAIN --> avg_loss=0.4233781695365906	acc=0.8027456402778625
[10/100] VALID RESULTS
{'loss': tensor(0.4595), 'acc': tensor(0.7792), 'pre': tensor(0.8312), 'rec': tensor(0.7018), 'f1': tensor(0.7610), 'confmat': [[tensor(640), tensor(130)], [tensor(272), tensor(779)]]}
TRAIN --> avg_loss=0.42066434025764465	acc=0.8075144290924072
[11/100] VALID RESULTS
{'loss': tensor(0.4706), 'acc': tensor(0.7507), 'pre': tensor(0.8449), 'rec': tensor(0.6151), 'f1': tensor(0.7119), 'confmat': [[tensor(561), tensor(103)], [tensor(351), tensor(806)]]}
TRAIN --> avg_loss=0.41612884402275085	acc=0.8086705207824707
[12/100] VALID RESULTS
{'loss': tensor(0.4643), 'acc': tensor(0.7650), 'pre': tensor(0.8408), 'rec': tensor(0.6546), 'f1': tensor(0.7361), 'confmat': [[tensor(597), tensor(113)], [tensor(315), tensor(796)]]}
TRAIN --> avg_loss=0.4123433828353882	acc=0.8108381628990173
[13/100] VALID RESULTS
{'loss': tensor(0.4779), 'acc': tensor(0.7446), 'pre': tensor(0.8498), 'rec': tensor(0.5954), 'f1': tensor(0.7002), 'confmat': [[tensor(543), tensor(96)], [tensor(369), tensor(813)]]}
TRAIN --> avg_loss=0.4114260673522949	acc=0.8108381628990173
[14/100] VALID RESULTS
{'loss': tensor(0.4631), 'acc': tensor(0.7573), 'pre': tensor(0.8348), 'rec': tensor(0.6425), 'f1': tensor(0.7261), 'confmat': [[tensor(586), tensor(116)], [tensor(326), tensor(793)]]}
TRAIN --> avg_loss=0.4036085903644562	acc=0.8125722408294678
[15/100] VALID RESULTS
{'loss': tensor(0.4578), 'acc': tensor(0.7781), 'pre': tensor(0.8256), 'rec': tensor(0.7061), 'f1': tensor(0.7612), 'confmat': [[tensor(644), tensor(136)], [tensor(268), tensor(773)]]}
TRAIN --> avg_loss=0.4016202390193939	acc=0.8135837912559509
[16/100] VALID RESULTS
{'loss': tensor(0.4673), 'acc': tensor(0.7578), 'pre': tensor(0.8428), 'rec': tensor(0.6349), 'f1': tensor(0.7242), 'confmat': [[tensor(579), tensor(108)], [tensor(333), tensor(801)]]}
TRAIN --> avg_loss=0.40058594942092896	acc=0.8153179287910461
[17/100] VALID RESULTS
{'loss': tensor(0.4507), 'acc': tensor(0.7781), 'pre': tensor(0.8144), 'rec': tensor(0.7215), 'f1': tensor(0.7651), 'confmat': [[tensor(658), tensor(150)], [tensor(254), tensor(759)]]}
TRAIN --> avg_loss=0.3949504792690277	acc=0.8173410296440125
[18/100] VALID RESULTS
{'loss': tensor(0.4586), 'acc': tensor(0.7770), 'pre': tensor(0.7998), 'rec': tensor(0.7401), 'f1': tensor(0.7688), 'confmat': [[tensor(675), tensor(169)], [tensor(237), tensor(740)]]}
TRAIN --> avg_loss=0.39170950651168823	acc=0.8205202221870422
[19/100] VALID RESULTS
{'loss': tensor(0.4593), 'acc': tensor(0.7688), 'pre': tensor(0.8235), 'rec': tensor(0.6853), 'f1': tensor(0.7481), 'confmat': [[tensor(625), tensor(134)], [tensor(287), tensor(775)]]}
TRAIN --> avg_loss=0.38966578245162964	acc=0.8161849975585938
[20/100] VALID RESULTS
{'loss': tensor(0.4576), 'acc': tensor(0.7836), 'pre': tensor(0.8136), 'rec': tensor(0.7368), 'f1': tensor(0.7733), 'confmat': [[tensor(672), tensor(154)], [tensor(240), tensor(755)]]}
TRAIN --> avg_loss=0.3857005536556244	acc=0.8215317726135254
[21/100] VALID RESULTS
{'loss': tensor(0.4569), 'acc': tensor(0.7743), 'pre': tensor(0.8096), 'rec': tensor(0.7182), 'f1': tensor(0.7612), 'confmat': [[tensor(655), tensor(154)], [tensor(257), tensor(755)]]}
TRAIN --> avg_loss=0.3769705593585968	acc=0.8261560797691345
[22/100] VALID RESULTS
{'loss': tensor(0.4617), 'acc': tensor(0.7825), 'pre': tensor(0.8057), 'rec': tensor(0.7456), 'f1': tensor(0.7745), 'confmat': [[tensor(680), tensor(164)], [tensor(232), tensor(745)]]}
TRAIN --> avg_loss=0.3744703531265259	acc=0.8303468227386475
[23/100] VALID RESULTS
{'loss': tensor(0.4705), 'acc': tensor(0.7573), 'pre': tensor(0.8446), 'rec': tensor(0.6316), 'f1': tensor(0.7227), 'confmat': [[tensor(576), tensor(106)], [tensor(336), tensor(803)]]}
TRAIN --> avg_loss=0.37020036578178406	acc=0.8335260152816772
[24/100] VALID RESULTS
{'loss': tensor(0.4652), 'acc': tensor(0.7606), 'pre': tensor(0.8362), 'rec': tensor(0.6491), 'f1': tensor(0.7309), 'confmat': [[tensor(592), tensor(116)], [tensor(320), tensor(793)]]}
TRAIN --> avg_loss=0.36613792181015015	acc=0.8343930840492249
[25/100] VALID RESULTS
{'loss': tensor(0.4620), 'acc': tensor(0.7727), 'pre': tensor(0.8276), 'rec': tensor(0.6897), 'f1': tensor(0.7524), 'confmat': [[tensor(629), tensor(131)], [tensor(283), tensor(778)]]}
TRAIN --> avg_loss=0.3601459562778473	acc=0.8355491161346436
[26/100] VALID RESULTS
{'loss': tensor(0.4702), 'acc': tensor(0.7666), 'pre': tensor(0.8313), 'rec': tensor(0.6700), 'f1': tensor(0.7420), 'confmat': [[tensor(611), tensor(124)], [tensor(301), tensor(785)]]}
TRAIN --> avg_loss=0.35633406043052673	acc=0.8385838270187378
[27/100] VALID RESULTS
{'loss': tensor(0.4705), 'acc': tensor(0.7622), 'pre': tensor(0.8368), 'rec': tensor(0.6524), 'f1': tensor(0.7332), 'confmat': [[tensor(595), tensor(116)], [tensor(317), tensor(793)]]}
TRAIN --> avg_loss=0.35110291838645935	acc=0.8419075012207031
[28/100] VALID RESULTS
{'loss': tensor(0.4740), 'acc': tensor(0.7655), 'pre': tensor(0.8178), 'rec': tensor(0.6842), 'f1': tensor(0.7451), 'confmat': [[tensor(624), tensor(139)], [tensor(288), tensor(770)]]}
TRAIN --> avg_loss=0.34451213479042053	acc=0.8446531891822815
[29/100] VALID RESULTS
{'loss': tensor(0.4745), 'acc': tensor(0.7600), 'pre': tensor(0.8258), 'rec': tensor(0.6601), 'f1': tensor(0.7337), 'confmat': [[tensor(602), tensor(127)], [tensor(310), tensor(782)]]}
TRAIN --> avg_loss=0.33967483043670654	acc=0.8521676063537598
[30/100] VALID RESULTS
{'loss': tensor(0.4906), 'acc': tensor(0.7512), 'pre': tensor(0.8255), 'rec': tensor(0.6382), 'f1': tensor(0.7199), 'confmat': [[tensor(582), tensor(123)], [tensor(330), tensor(786)]]}
TRAIN --> avg_loss=0.3347618877887726	acc=0.8502890467643738
[31/100] VALID RESULTS
{'loss': tensor(0.4828), 'acc': tensor(0.7831), 'pre': tensor(0.7724), 'rec': tensor(0.8037), 'f1': tensor(0.7877), 'confmat': [[tensor(733), tensor(216)], [tensor(179), tensor(693)]]}
TRAIN --> avg_loss=0.3274429440498352	acc=0.8599711060523987
[32/100] VALID RESULTS
{'loss': tensor(0.4791), 'acc': tensor(0.7688), 'pre': tensor(0.8209), 'rec': tensor(0.6886), 'f1': tensor(0.7490), 'confmat': [[tensor(628), tensor(137)], [tensor(284), tensor(772)]]}
TRAIN --> avg_loss=0.3226569592952728	acc=0.8601155877113342
[33/100] VALID RESULTS
{'loss': tensor(0.4853), 'acc': tensor(0.7600), 'pre': tensor(0.8137), 'rec': tensor(0.6754), 'f1': tensor(0.7382), 'confmat': [[tensor(616), tensor(141)], [tensor(296), tensor(768)]]}
TRAIN --> avg_loss=0.31568771600723267	acc=0.8673410415649414
[34/100] VALID RESULTS
{'loss': tensor(0.4842), 'acc': tensor(0.7655), 'pre': tensor(0.8074), 'rec': tensor(0.6985), 'f1': tensor(0.7490), 'confmat': [[tensor(637), tensor(152)], [tensor(275), tensor(757)]]}
TRAIN --> avg_loss=0.3096679151058197	acc=0.8690751194953918
[35/100] VALID RESULTS
{'loss': tensor(0.4827), 'acc': tensor(0.7683), 'pre': tensor(0.8025), 'rec': tensor(0.7127), 'f1': tensor(0.7549), 'confmat': [[tensor(650), tensor(160)], [tensor(262), tensor(749)]]}
TRAIN --> avg_loss=0.3037351369857788	acc=0.8684971332550049
[36/100] VALID RESULTS
{'loss': tensor(0.4975), 'acc': tensor(0.7677), 'pre': tensor(0.8131), 'rec': tensor(0.6963), 'f1': tensor(0.7501), 'confmat': [[tensor(635), tensor(146)], [tensor(277), tensor(763)]]}
TRAIN --> avg_loss=0.2983926236629486	acc=0.8757225275039673
[37/100] VALID RESULTS
{'loss': tensor(0.5050), 'acc': tensor(0.7666), 'pre': tensor(0.8078), 'rec': tensor(0.7007), 'f1': tensor(0.7504), 'confmat': [[tensor(639), tensor(152)], [tensor(273), tensor(757)]]}
EARLY STOPPING
[finito] TEST RESULTS
{'loss': tensor(0.4694), 'acc': tensor(0.7867), 'pre': tensor(0.7995), 'rec': tensor(0.7547), 'f1': tensor(0.7764), 'confmat': [[tensor(323), tensor(81)], [tensor(105), tensor(363)]]}
```
</details>

## 3. zadatak (25%)

✓


## 4. zadatak (25%)

**! NB: Svi rezultati ispod imaju neznatno neispravan izračun točnosti/opoziva/preciznost/f1 mjere. Koristio sam (logits > 0.5) umjesto (logits > 0). To znači da sam razrede pridodijeljivao na sljedeći način 0%-61% --> 0, 62%-100% --> 1, iako je model učio s standradnih 50/50 u funkciji gubitka. Vrijednosti funkcije gubitka su ispravne. Eksperimente neću ponavljati. Ispod doneseni zaključci su možda neispravni.**

U svim eksperimentima u ovom zadatku koristim `batch_size=10`, `max_epochs=300`, `learning_rate=1e-4`, `clip=0.25`, `early_stopping_epsilon=1e-7`, `early_stopping_patince=20`. Te hiperparametre nisam nigdje mijenjao.


> Neovisno o tome koju RNN ćeliju ste odabrali u trećem zadatku, proširite vaš kod na način da vrsta RNN ćelije bude argument. Pokrenite vaš kod za preostale vrste RNN ćelija i zapišite rezultate. Je li neka ćelija očiti pobjednik? Je li neka ćelija očiti gubitnik?

<details>

<summary> All results </summary>

```js
{
    'GRU': [{
        'loss': tensor(0.4391, device = 'cuda:0'),
        'acc': tensor(0.8108, device = 'cuda:0'),
        'pre': tensor(0.8247, device = 'cuda:0'),
        'rec': tensor(0.7804, device = 'cuda:0'),
        'f1': tensor(0.8019, device = 'cuda:0'),
        'confmat': [
            [tensor(334, device = 'cuda:0'), tensor(71, device = 'cuda:0')],
            [tensor(94, device = 'cuda:0'), tensor(373, device = 'cuda:0')]
        ]
    }, {
        'loss': tensor(0.4701, device = 'cuda:0'),
        'acc': tensor(0.7615, device = 'cuda:0'),
        'pre': tensor(0.8741, device = 'cuda:0'),
        'rec': tensor(0.6005, device = 'cuda:0'),
        'f1': tensor(0.7119, device = 'cuda:0'),
        'confmat': [
            [tensor(257, device = 'cuda:0'), tensor(37, device = 'cuda:0')],
            [tensor(171, device = 'cuda:0'), tensor(407, device = 'cuda:0')]
        ]
    }, {
        'loss': tensor(0.4503, device = 'cuda:0'),
        'acc': tensor(0.7936, device = 'cuda:0'),
        'pre': tensor(0.8523, device = 'cuda:0'),
        'rec': tensor(0.7009, device = 'cuda:0'),
        'f1': tensor(0.7692, device = 'cuda:0'),
        'confmat': [
            [tensor(300, device = 'cuda:0'), tensor(52, device = 'cuda:0')],
            [tensor(128, device = 'cuda:0'), tensor(392, device = 'cuda:0')]
        ]
    }, {
        'loss': tensor(0.4421, device = 'cuda:0'),
        'acc': tensor(0.7924, device = 'cuda:0'),
        'pre': tensor(0.7839, device = 'cuda:0'),
        'rec': tensor(0.7967, device = 'cuda:0'),
        'f1': tensor(0.7903, device = 'cuda:0'),
        'confmat': [
            [tensor(341, device = 'cuda:0'), tensor(94, device = 'cuda:0')],
            [tensor(87, device = 'cuda:0'), tensor(350, device = 'cuda:0')]
        ]
    }, {
        'loss': tensor(0.4303, device = 'cuda:0'),
        'acc': tensor(0.8154, device = 'cuda:0'),
        'pre': tensor(0.8201, device = 'cuda:0'),
        'rec': tensor(0.7991, device = 'cuda:0'),
        'f1': tensor(0.8095, device = 'cuda:0'),
        'confmat': [
            [tensor(342, device = 'cuda:0'), tensor(75, device = 'cuda:0')],
            [tensor(86, device = 'cuda:0'), tensor(369, device = 'cuda:0')]
        ]
    }, {
        'loss': tensor(0.4522, device = 'cuda:0'),
        'acc': tensor(0.8096, device = 'cuda:0'),
        'pre': tensor(0.8005, device = 'cuda:0'),
        'rec': tensor(0.8154, device = 'cuda:0'),
        'f1': tensor(0.8079, device = 'cuda:0'),
        'confmat': [
            [tensor(349, device = 'cuda:0'), tensor(87, device = 'cuda:0')],
            [tensor(79, device = 'cuda:0'), tensor(357, device = 'cuda:0')]
        ]
    }, {
        'loss': tensor(0.4318, device = 'cuda:0'),
        'acc': tensor(0.8073, device = 'cuda:0'),
        'pre': tensor(0.8385, device = 'cuda:0'),
        'rec': tensor(0.7523, device = 'cuda:0'),
        'f1': tensor(0.7931, device = 'cuda:0'),
        'confmat': [
            [tensor(322, device = 'cuda:0'), tensor(62, device = 'cuda:0')],
            [tensor(106, device = 'cuda:0'), tensor(382, device = 'cuda:0')]
        ]
    }, {
        'loss': tensor(0.4576, device = 'cuda:0'),
        'acc': tensor(0.7959, device = 'cuda:0'),
        'pre': tensor(0.8788, device = 'cuda:0'),
        'rec': tensor(0.6776, device = 'cuda:0'),
        'f1': tensor(0.7652, device = 'cuda:0'),
        'confmat': [
            [tensor(290, device = 'cuda:0'), tensor(40, device = 'cuda:0')],
            [tensor(138, device = 'cuda:0'), tensor(404, device = 'cuda:0')]
        ]
    }, {
        'loss': tensor(0.4264, device = 'cuda:0'),
        'acc': tensor(0.8050, device = 'cuda:0'),
        'pre': tensor(0.8258, device = 'cuda:0'),
        'rec': tensor(0.7640, device = 'cuda:0'),
        'f1': tensor(0.7937, device = 'cuda:0'),
        'confmat': [
            [tensor(327, device = 'cuda:0'), tensor(69, device = 'cuda:0')],
            [tensor(101, device = 'cuda:0'), tensor(375, device = 'cuda:0')]
        ]
    }, {
        'loss': tensor(0.4368, device = 'cuda:0'),
        'acc': tensor(0.8073, device = 'cuda:0'),
        'pre': tensor(0.8066, device = 'cuda:0'),
        'rec': tensor(0.7991, device = 'cuda:0'),
        'f1': tensor(0.8028, device = 'cuda:0'),
        'confmat': [
            [tensor(342, device = 'cuda:0'), tensor(82, device = 'cuda:0')],
            [tensor(86, device = 'cuda:0'), tensor(362, device = 'cuda:0')]
        ]
    }],
    'LSTM': [{
        'loss': tensor(0.4654, device = 'cuda:0'),
        'acc': tensor(0.7683, device = 'cuda:0'),
        'pre': tensor(0.8792, device = 'cuda:0'),
        'rec': tensor(0.6121, device = 'cuda:0'),
        'f1': tensor(0.7218, device = 'cuda:0'),
        'confmat': [
            [tensor(262, device = 'cuda:0'), tensor(36, device = 'cuda:0')],
            [tensor(166, device = 'cuda:0'), tensor(408, device = 'cuda:0')]
        ]
    }, {
        'loss': tensor(0.4525, device = 'cuda:0'),
        'acc': tensor(0.7970, device = 'cuda:0'),
        'pre': tensor(0.8277, device = 'cuda:0'),
        'rec': tensor(0.7407, device = 'cuda:0'),
        'f1': tensor(0.7818, device = 'cuda:0'),
        'confmat': [
            [tensor(317, device = 'cuda:0'), tensor(66, device = 'cuda:0')],
            [tensor(111, device = 'cuda:0'), tensor(378, device = 'cuda:0')]
        ]
    }, {
        'loss': tensor(0.4231, device = 'cuda:0'),
        'acc': tensor(0.7959, device = 'cuda:0'),
        'pre': tensor(0.8255, device = 'cuda:0'),
        'rec': tensor(0.7407, device = 'cuda:0'),
        'f1': tensor(0.7808, device = 'cuda:0'),
        'confmat': [
            [tensor(317, device = 'cuda:0'), tensor(67, device = 'cuda:0')],
            [tensor(111, device = 'cuda:0'), tensor(377, device = 'cuda:0')]
        ]
    }, {
        'loss': tensor(0.4411, device = 'cuda:0'),
        'acc': tensor(0.7970, device = 'cuda:0'),
        'pre': tensor(0.7912, device = 'cuda:0'),
        'rec': tensor(0.7967, device = 'cuda:0'),
        'f1': tensor(0.7939, device = 'cuda:0'),
        'confmat': [
            [tensor(341, device = 'cuda:0'), tensor(90, device = 'cuda:0')],
            [tensor(87, device = 'cuda:0'), tensor(354, device = 'cuda:0')]
        ]
    }, {
        'loss': tensor(0.4412, device = 'cuda:0'),
        'acc': tensor(0.8108, device = 'cuda:0'),
        'pre': tensor(0.8263, device = 'cuda:0'),
        'rec': tensor(0.7780, device = 'cuda:0'),
        'f1': tensor(0.8014, device = 'cuda:0'),
        'confmat': [
            [tensor(333, device = 'cuda:0'), tensor(70, device = 'cuda:0')],
            [tensor(95, device = 'cuda:0'), tensor(374, device = 'cuda:0')]
        ]
    }, {
        'loss': tensor(0.4514, device = 'cuda:0'),
        'acc': tensor(0.7959, device = 'cuda:0'),
        'pre': tensor(0.8189, device = 'cuda:0'),
        'rec': tensor(0.7500, device = 'cuda:0'),
        'f1': tensor(0.7829, device = 'cuda:0'),
        'confmat': [
            [tensor(321, device = 'cuda:0'), tensor(71, device = 'cuda:0')],
            [tensor(107, device = 'cuda:0'), tensor(373, device = 'cuda:0')]
        ]
    }, {
        'loss': tensor(0.4335, device = 'cuda:0'),
        'acc': tensor(0.8028, device = 'cuda:0'),
        'pre': tensor(0.7963, device = 'cuda:0'),
        'rec': tensor(0.8037, device = 'cuda:0'),
        'f1': tensor(0.8000, device = 'cuda:0'),
        'confmat': [
            [tensor(344, device = 'cuda:0'), tensor(88, device = 'cuda:0')],
            [tensor(84, device = 'cuda:0'), tensor(356, device = 'cuda:0')]
        ]
    }, {
        'loss': tensor(0.4927, device = 'cuda:0'),
        'acc': tensor(0.7764, device = 'cuda:0'),
        'pre': tensor(0.8923, device = 'cuda:0'),
        'rec': tensor(0.6192, device = 'cuda:0'),
        'f1': tensor(0.7310, device = 'cuda:0'),
        'confmat': [
            [tensor(265, device = 'cuda:0'), tensor(32, device = 'cuda:0')],
            [tensor(163, device = 'cuda:0'), tensor(412, device = 'cuda:0')]
        ]
    }, {
        'loss': tensor(0.4815, device = 'cuda:0'),
        'acc': tensor(0.7718, device = 'cuda:0'),
        'pre': tensor(0.8962, device = 'cuda:0'),
        'rec': tensor(0.6051, device = 'cuda:0'),
        'f1': tensor(0.7225, device = 'cuda:0'),
        'confmat': [
            [tensor(259, device = 'cuda:0'), tensor(30, device = 'cuda:0')],
            [tensor(169, device = 'cuda:0'), tensor(414, device = 'cuda:0')]
        ]
    }, {
        'loss': tensor(0.4634, device = 'cuda:0'),
        'acc': tensor(0.7936, device = 'cuda:0'),
        'pre': tensor(0.8605, device = 'cuda:0'),
        'rec': tensor(0.6916, device = 'cuda:0'),
        'f1': tensor(0.7668, device = 'cuda:0'),
        'confmat': [
            [tensor(296, device = 'cuda:0'), tensor(48, device = 'cuda:0')],
            [tensor(132, device = 'cuda:0'), tensor(396, device = 'cuda:0')]
        ]
    }],
    'RNN': [{
        'loss': tensor(0.4787, device = 'cuda:0'),
        'acc': tensor(0.7706, device = 'cuda:0'),
        'pre': tensor(0.7436, device = 'cuda:0'),
        'rec': tensor(0.8131, device = 'cuda:0'),
        'f1': tensor(0.7768, device = 'cuda:0'),
        'confmat': [
            [tensor(348, device = 'cuda:0'), tensor(120, device = 'cuda:0')],
            [tensor(80, device = 'cuda:0'), tensor(324, device = 'cuda:0')]
        ]
    }, {
        'loss': tensor(0.4597, device = 'cuda:0'),
        'acc': tensor(0.7833, device = 'cuda:0'),
        'pre': tensor(0.7559, device = 'cuda:0'),
        'rec': tensor(0.8248, device = 'cuda:0'),
        'f1': tensor(0.7888, device = 'cuda:0'),
        'confmat': [
            [tensor(353, device = 'cuda:0'), tensor(114, device = 'cuda:0')],
            [tensor(75, device = 'cuda:0'), tensor(330, device = 'cuda:0')]
        ]
    }, {
        'loss': tensor(0.4580, device = 'cuda:0'),
        'acc': tensor(0.7867, device = 'cuda:0'),
        'pre': tensor(0.7909, device = 'cuda:0'),
        'rec': tensor(0.7687, device = 'cuda:0'),
        'f1': tensor(0.7796, device = 'cuda:0'),
        'confmat': [
            [tensor(329, device = 'cuda:0'), tensor(87, device = 'cuda:0')],
            [tensor(99, device = 'cuda:0'), tensor(357, device = 'cuda:0')]
        ]
    }, {
        'loss': tensor(0.4583, device = 'cuda:0'),
        'acc': tensor(0.7844, device = 'cuda:0'),
        'pre': tensor(0.7740, device = 'cuda:0'),
        'rec': tensor(0.7921, device = 'cuda:0'),
        'f1': tensor(0.7829, device = 'cuda:0'),
        'confmat': [
            [tensor(339, device = 'cuda:0'), tensor(99, device = 'cuda:0')],
            [tensor(89, device = 'cuda:0'), tensor(345, device = 'cuda:0')]
        ]
    }, {
        'loss': tensor(0.4521, device = 'cuda:0'),
        'acc': tensor(0.7970, device = 'cuda:0'),
        'pre': tensor(0.7758, device = 'cuda:0'),
        'rec': tensor(0.8248, device = 'cuda:0'),
        'f1': tensor(0.7995, device = 'cuda:0'),
        'confmat': [
            [tensor(353, device = 'cuda:0'), tensor(102, device = 'cuda:0')],
            [tensor(75, device = 'cuda:0'), tensor(342, device = 'cuda:0')]
        ]
    }, {
        'loss': tensor(0.4625, device = 'cuda:0'),
        'acc': tensor(0.7890, device = 'cuda:0'),
        'pre': tensor(0.8194, device = 'cuda:0'),
        'rec': tensor(0.7313, device = 'cuda:0'),
        'f1': tensor(0.7728, device = 'cuda:0'),
        'confmat': [
            [tensor(313, device = 'cuda:0'), tensor(69, device = 'cuda:0')],
            [tensor(115, device = 'cuda:0'), tensor(375, device = 'cuda:0')]
        ]
    }, {
        'loss': tensor(0.4990, device = 'cuda:0'),
        'acc': tensor(0.7775, device = 'cuda:0'),
        'pre': tensor(0.8382, device = 'cuda:0'),
        'rec': tensor(0.6776, device = 'cuda:0'),
        'f1': tensor(0.7494, device = 'cuda:0'),
        'confmat': [
            [tensor(290, device = 'cuda:0'), tensor(56, device = 'cuda:0')],
            [tensor(138, device = 'cuda:0'), tensor(388, device = 'cuda:0')]
        ]
    }, {
        'loss': tensor(0.5023, device = 'cuda:0'),
        'acc': tensor(0.7695, device = 'cuda:0'),
        'pre': tensor(0.8271, device = 'cuda:0'),
        'rec': tensor(0.6706, device = 'cuda:0'),
        'f1': tensor(0.7406, device = 'cuda:0'),
        'confmat': [
            [tensor(287, device = 'cuda:0'), tensor(60, device = 'cuda:0')],
            [tensor(141, device = 'cuda:0'), tensor(384, device = 'cuda:0')]
        ]
    }, {
        'loss': tensor(0.5075, device = 'cuda:0'),
        'acc': tensor(0.7752, device = 'cuda:0'),
        'pre': tensor(0.7915, device = 'cuda:0'),
        'rec': tensor(0.7360, device = 'cuda:0'),
        'f1': tensor(0.7627, device = 'cuda:0'),
        'confmat': [
            [tensor(315, device = 'cuda:0'), tensor(83, device = 'cuda:0')],
            [tensor(113, device = 'cuda:0'), tensor(361, device = 'cuda:0')]
        ]
    }, {
        'loss': tensor(0.4509, device = 'cuda:0'),
        'acc': tensor(0.7913, device = 'cuda:0'),
        'pre': tensor(0.8060, device = 'cuda:0'),
        'rec': tensor(0.7570, device = 'cuda:0'),
        'f1': tensor(0.7807, device = 'cuda:0'),
        'confmat': [
            [tensor(324, device = 'cuda:0'), tensor(78, device = 'cuda:0')],
            [tensor(104, device = 'cuda:0'), tensor(366, device = 'cuda:0')]
        ]
    }],
    'BASELINE': [{
        'loss': tensor(0.4946, device = 'cuda:0'),
        'acc': tensor(0.7443, device = 'cuda:0'),
        'pre': tensor(0.8428, device = 'cuda:0'),
        'rec': tensor(0.5888, device = 'cuda:0'),
        'f1': tensor(0.6933, device = 'cuda:0'),
        'confmat': [
            [tensor(252, device = 'cuda:0'), tensor(47, device = 'cuda:0')],
            [tensor(176, device = 'cuda:0'), tensor(397, device = 'cuda:0')]
        ]
    }, {
        'loss': tensor(0.4820, device = 'cuda:0'),
        'acc': tensor(0.7569, device = 'cuda:0'),
        'pre': tensor(0.8396, device = 'cuda:0'),
        'rec': tensor(0.6238, device = 'cuda:0'),
        'f1': tensor(0.7158, device = 'cuda:0'),
        'confmat': [
            [tensor(267, device = 'cuda:0'), tensor(51, device = 'cuda:0')],
            [tensor(161, device = 'cuda:0'), tensor(393, device = 'cuda:0')]
        ]
    }, {
        'loss': tensor(0.4804, device = 'cuda:0'),
        'acc': tensor(0.7649, device = 'cuda:0'),
        'pre': tensor(0.8232, device = 'cuda:0'),
        'rec': tensor(0.6636, device = 'cuda:0'),
        'f1': tensor(0.7348, device = 'cuda:0'),
        'confmat': [
            [tensor(284, device = 'cuda:0'), tensor(61, device = 'cuda:0')],
            [tensor(144, device = 'cuda:0'), tensor(383, device = 'cuda:0')]
        ]
    }, {
        'loss': tensor(0.4791, device = 'cuda:0'),
        'acc': tensor(0.7706, device = 'cuda:0'),
        'pre': tensor(0.8032, device = 'cuda:0'),
        'rec': tensor(0.7056, device = 'cuda:0'),
        'f1': tensor(0.7512, device = 'cuda:0'),
        'confmat': [
            [tensor(302, device = 'cuda:0'), tensor(74, device = 'cuda:0')],
            [tensor(126, device = 'cuda:0'), tensor(370, device = 'cuda:0')]
        ]
    }, {
        'loss': tensor(0.4776, device = 'cuda:0'),
        'acc': tensor(0.7569, device = 'cuda:0'),
        'pre': tensor(0.8103, device = 'cuda:0'),
        'rec': tensor(0.6589, device = 'cuda:0'),
        'f1': tensor(0.7268, device = 'cuda:0'),
        'confmat': [
            [tensor(282, device = 'cuda:0'), tensor(66, device = 'cuda:0')],
            [tensor(146, device = 'cuda:0'), tensor(378, device = 'cuda:0')]
        ]
    }, {
        'loss': tensor(0.4797, device = 'cuda:0'),
        'acc': tensor(0.7626, device = 'cuda:0'),
        'pre': tensor(0.8318, device = 'cuda:0'),
        'rec': tensor(0.6472, device = 'cuda:0'),
        'f1': tensor(0.7280, device = 'cuda:0'),
        'confmat': [
            [tensor(277, device = 'cuda:0'), tensor(56, device = 'cuda:0')],
            [tensor(151, device = 'cuda:0'), tensor(388, device = 'cuda:0')]
        ]
    }, {
        'loss': tensor(0.4834, device = 'cuda:0'),
        'acc': tensor(0.7626, device = 'cuda:0'),
        'pre': tensor(0.8113, device = 'cuda:0'),
        'rec': tensor(0.6729, device = 'cuda:0'),
        'f1': tensor(0.7356, device = 'cuda:0'),
        'confmat': [
            [tensor(288, device = 'cuda:0'), tensor(67, device = 'cuda:0')],
            [tensor(140, device = 'cuda:0'), tensor(377, device = 'cuda:0')]
        ]
    }, {
        'loss': tensor(0.4852, device = 'cuda:0'),
        'acc': tensor(0.7546, device = 'cuda:0'),
        'pre': tensor(0.8429, device = 'cuda:0'),
        'rec': tensor(0.6145, device = 'cuda:0'),
        'f1': tensor(0.7108, device = 'cuda:0'),
        'confmat': [
            [tensor(263, device = 'cuda:0'), tensor(49, device = 'cuda:0')],
            [tensor(165, device = 'cuda:0'), tensor(395, device = 'cuda:0')]
        ]
    }, {
        'loss': tensor(0.4932, device = 'cuda:0'),
        'acc': tensor(0.7500, device = 'cuda:0'),
        'pre': tensor(0.8431, device = 'cuda:0'),
        'rec': tensor(0.6028, device = 'cuda:0'),
        'f1': tensor(0.7030, device = 'cuda:0'),
        'confmat': [
            [tensor(258, device = 'cuda:0'), tensor(48, device = 'cuda:0')],
            [tensor(170, device = 'cuda:0'), tensor(396, device = 'cuda:0')]
        ]
    }, {
        'loss': tensor(0.4799, device = 'cuda:0'),
        'acc': tensor(0.7706, device = 'cuda:0'),
        'pre': tensor(0.8132, device = 'cuda:0'),
        'rec': tensor(0.6916, device = 'cuda:0'),
        'f1': tensor(0.7475, device = 'cuda:0'),
        'confmat': [
            [tensor(296, device = 'cuda:0'), tensor(68, device = 'cuda:0')],
            [tensor(132, device = 'cuda:0'), tensor(376, device = 'cuda:0')]
        ]
    }]
}
```

</details>

```js
# ~~~~~~ Results summary with second momentum measures ~~~~~~ #
[GRU] [LOSS] avg:0.44366335570812226 std:0.013043080857846681
[GRU] [ACC] avg:0.7998853087425232 std:0.014759760525249644
[GRU] [PRE] avg:0.8305315375328064 std:0.029297644967659985
[GRU] [REC] avg:0.7485981285572052 std:0.06511276290406433
[GRU] [F1] avg:0.784544849395752 std:0.0280643815152603
[LSTM] [LOSS] avg:0.45457369089126587 std:0.02046703032747012
[LSTM] [ACC] avg:0.7909403681755066 std:0.013235850220831734
[LSTM] [PRE] avg:0.8413966298103333 std:0.036210948594105015
[LSTM] [REC] avg:0.7137850463390351 std:0.0730994790160269
[LSTM] [F1] avg:0.7682947814464569 std:0.029950478133174762
[RNN] [LOSS] avg:0.47290125489234924 std:0.021000310176263103
[RNN] [ACC] avg:0.7824541330337524 std:0.008597858191131649
[RNN] [PRE] avg:0.7922179281711579 std:0.02922466869642889
[RNN] [REC] avg:0.7595794379711152 std:0.053396406221510186
[RNN] [F1] avg:0.7733963608741761 std:0.01697504648779906
[BASELINE] [LOSS] avg:0.48349606096744535 std:0.005607380672199687
[BASELINE] [ACC] avg:0.7594036757946014 std:0.008105784084835586
[BASELINE] [PRE] avg:0.8261528968811035 std:0.014957578308976155
[BASELINE] [REC] avg:0.6469626128673553 std:0.036667353272792996
[BASELINE] [F1] avg:0.7246829748153687 std:0.017864219848154114
```

GRU je očiti pobjednik gledano po točnosti i f1 mjeri. Nije očito tko je gubitnik -- LSTM ili RNN. Ali sve povratne neuronske mreže su očigledno bolje od baselinea.

### Eksperiment 1

> Ponovite ovu usporedbu uz izmjenu hiperparametara povratnih neuronskih mreža. Idući hiperparametri povratnih neuronskih mreža su nam interesantni:
>
> - hidden_size
> - num_layers
> - dropout: primjenjen između uzastopnih slojeva RNNa (funkcionira samo za 2+ slojeva)
> - bidirectional: dimenzionalnost izlaza dvosmjerne rnn ćelije je dvostruka
>
> Isprobajte barem 3 različite vrijednosti za svaki hiperparametar

Testirao sam sljedeće konfiguracije i dotične modele pokrenuo samo jedanput:

- hidden_size $\in \{ 50, 150, 300 \}$
- num_layers $\in \{ 2, 3, 5 \}$
- dropout $\in \{ 0, 0.5, 0.9 \}$
- rnn_cell_type $\in \{ \mathrm{GRU}, \mathrm{LSTM}, \mathrm{RNN}\}$

S više računalnih resursa i efikasnijim modelom bi mogao pokrenuti zanimljivije konfiguracije, na primjer zanimljivo bi bilo vidjeti što se događa kada se stavi hidden_size na 1000 ili 10000, ili što se dogodi kada se broj slojeva stavi na 20. Uzeo sam gore spomenute konfiguracije jer smatram da sadrže esencijalne set parametara koji imaju šansu da razumno poboljša performansu i mislim da je prioritet njih testirati prije postavljanja parametara na interesantne ali nerazumne vrijednosti.

Ovo daje $3^4=81$ različitih kombinacija. Rezultati njihovog pokretanja sortirani po točnosti su dani ispod. Sveukupni rezultati s dodatnim komentarima su u excelici `./results.xlsx`.

<details>

<summary> Eksperiment 1 -- svi rezultati </summary>

```js
run_name	metric	value	stddev
LSTM_n=5_h=300_d=0.5__mvs=-1_mtiv=0_lr=0.0001_w2v=g	ACC	0.8234000206	0
GRU_n=5_h=150_d=0.5__mvs=-1_mtiv=0_lr=0.0001_w2v=g	ACC	0.8187999725	0
GRU_n=2_h=150_d=0.5__mvs=-1_mtiv=0_lr=0.0001_w2v=g	ACC	0.8165000081	0
LSTM_n=5_h=150_d=0.5__mvs=-1_mtiv=0_lr=0.0001_w2v=g	ACC	0.8141999841	0
LSTM_n=2_h=300_d=0__mvs=-1_mtiv=0_lr=0.0001_w2v=g	ACC	0.8130999804	0
GRU_n=5_h=300_d=0__mvs=-1_mtiv=0_lr=0.0001_w2v=g	ACC	0.8126199961	0.008140366136
LSTM_n=2_h=50_d=0.9__mvs=-1_mtiv=0_lr=0.0001_w2v=g	ACC	0.8119000196	0
LSTM_n=3_h=150_d=0__mvs=-1_mtiv=0_lr=0.0001_w2v=g	ACC	0.8119000196	0
LSTM_n=3_h=150_d=0.5__mvs=-1_mtiv=0_lr=0.0001_w2v=g	ACC	0.8119000196	0
GRU_n=3_h=300_d=0__mvs=-1_mtiv=0_lr=0.0001_w2v=g	ACC	0.8112599969	0.008473624997
GRU_n=5_h=300_d=0.5__mvs=-1_mtiv=0_lr=0.0001_w2v=g	ACC	0.8108000159	0
LSTM_n=2_h=50_d=0.5__mvs=-1_mtiv=0_lr=0.0001_w2v=g	ACC	0.8108000159	0
LSTM_n=2_h=150_d=0.9__mvs=-1_mtiv=0_lr=0.0001_w2v=g	ACC	0.8095999956	0
LSTM_n=3_h=300_d=0__mvs=-1_mtiv=0_lr=0.0001_w2v=g	ACC	0.8084999919	0
LSTM_n=3_h=300_d=0.9__mvs=-1_mtiv=0_lr=0.0001_w2v=g	ACC	0.8084999919	0
GRU_n=3_h=300_d=0.5__mvs=-1_mtiv=0_lr=0.0001_w2v=g	ACC	0.8084800005	0.006360623258
GRU_n=2_h=150_d=0__mvs=-1_mtiv=0_lr=0.0001_w2v=g	ACC	0.807340014	0.008027854301
LSTM_n=2_h=300_d=0.9__mvs=-1_mtiv=0_lr=0.0001_w2v=g	ACC	0.8072999716	0
LSTM_n=5_h=150_d=0__mvs=-1_mtiv=0_lr=0.0001_w2v=g	ACC	0.8072999716	0
GRU_n=2_h=300_d=0.5__mvs=-1_mtiv=0_lr=0.0001_w2v=g	ACC	0.8071200013	0.01072631273
GRU_n=2_h=300_d=0.9__mvs=-1_mtiv=0_lr=0.0001_w2v=g	ACC	0.8069000125	0.006586658611
LSTM_n=5_h=50_d=0__mvs=-1_mtiv=0_lr=0.0001_w2v=g	ACC	0.8062000275	0
GRU_n=2_h=300_d=0__mvs=-1_mtiv=0_lr=0.0001_w2v=g	ACC	0.8057599902	0.009481470148
GRU_n=3_h=50_d=0__mvs=-1_mtiv=0_lr=0.0001_w2v=g	ACC	0.805279994	0.006930327868
GRU_n=5_h=50_d=0__mvs=-1_mtiv=0_lr=0.0001_w2v=g	ACC	0.8048200011	0.004984526546
LSTM_n=2_h=150_d=0__mvs=-1_mtiv=0_lr=0.0001_w2v=g	ACC	0.8039000034	0
LSTM_n=3_h=150_d=0.9__mvs=-1_mtiv=0_lr=0.0001_w2v=g	ACC	0.8039000034	0
GRU_n=3_h=50_d=0.5__mvs=-1_mtiv=0_lr=0.0001_w2v=g	ACC	0.8038999915	0.004398180002
GRU_n=3_h=150_d=0.5__mvs=-1_mtiv=0_lr=0.0001_w2v=g	ACC	0.8027999997	0
GRU_n=5_h=150_d=0.9__mvs=-1_mtiv=0_lr=0.0001_w2v=g	ACC	0.8027999997	0
LSTM_n=3_h=50_d=0__mvs=-1_mtiv=0_lr=0.0001_w2v=g	ACC	0.8027999997	0
LSTM_n=3_h=300_d=0.5__mvs=-1_mtiv=0_lr=0.0001_w2v=g	ACC	0.8015999794	0
GRU_n=2_h=50_d=0.5__mvs=-1_mtiv=0_lr=0.0001_w2v=g	ACC	0.8013999939	0.002844649289
GRU_n=3_h=300_d=0.9__mvs=-1_mtiv=0_lr=0.0001_w2v=g	ACC	0.8009200096	0.01300145099
LSTM_n=2_h=50_d=0__mvs=-1_mtiv=0_lr=0.0001_w2v=g	ACC	0.8004999757	0
LSTM_n=3_h=50_d=0.9__mvs=-1_mtiv=0_lr=0.0001_w2v=g	ACC	0.8004999757	0
LSTM_n=5_h=300_d=0.9__mvs=-1_mtiv=0_lr=0.0001_w2v=g	ACC	0.8004999757	0
LSTM_n=5_h=50_d=0.5__mvs=-1_mtiv=0_lr=0.0001_w2v=g	ACC	0.8004999757	0
GRU_n=2_h=150_d=0.9__mvs=-1_mtiv=0_lr=0.0001_w2v=g	ACC	0.799300015	0
GRU_n=5_h=50_d=0.5__mvs=-1_mtiv=0_lr=0.0001_w2v=g	ACC	0.799300015	0
RNN_n=5_h=50_d=0.5__mvs=-1_mtiv=0_lr=0.0001_w2v=g	ACC	0.799300015	0
LSTM_n=2_h=150_d=0.5__mvs=-1_mtiv=0_lr=0.0001_w2v=g	ACC	0.7982000113	0
RNN_n=2_h=150_d=0.9__mvs=-1_mtiv=0_lr=0.0001_w2v=g	ACC	0.7982000113	0
GRU_n=2_h=50_d=0__mvs=-1_mtiv=0_lr=0.0001_w2v=g	ACC	0.7961749882	0.004329194005
GRU_n=2_h=50_d=0.9__mvs=-1_mtiv=0_lr=0.0001_w2v=g	ACC	0.7961200118	0.007847390577
GRU_n=5_h=150_d=0__mvs=-1_mtiv=0_lr=0.0001_w2v=g	ACC	0.7958999872	0
RNN_n=3_h=150_d=0.5__mvs=-1_mtiv=0_lr=0.0001_w2v=g	ACC	0.7958999872	0
LSTM_n=2_h=300_d=0.5__mvs=-1_mtiv=0_lr=0.0001_w2v=g	ACC	0.7947000265	0
LSTM_n=5_h=150_d=0.9__mvs=-1_mtiv=0_lr=0.0001_w2v=g	ACC	0.7947000265	0
RNN_n=3_h=150_d=0.9__mvs=-1_mtiv=0_lr=0.0001_w2v=g	ACC	0.7947000265	0
GRU_n=3_h=150_d=0.9__mvs=-1_mtiv=0_lr=0.0001_w2v=g	ACC	0.7936000228	0
GRU_n=5_h=300_d=0.9__mvs=-1_mtiv=0_lr=0.0001_w2v=g	ACC	0.7936000228	0
GRU_n=3_h=50_d=0.9__mvs=-1_mtiv=0_lr=0.0001_w2v=g	ACC	0.7915400028	0.005637948568
GRU_n=3_h=150_d=0__mvs=-1_mtiv=0_lr=0.0001_w2v=g	ACC	0.7900999784	0
RNN_n=3_h=50_d=0.5__mvs=-1_mtiv=0_lr=0.0001_w2v=g	ACC	0.7900999784	0
RNN_n=5_h=150_d=0.5__mvs=-1_mtiv=0_lr=0.0001_w2v=g	ACC	0.7900999784	0
RNN_n=2_h=50_d=0.5__mvs=-1_mtiv=0_lr=0.0001_w2v=g	ACC	0.7889999747	0
LSTM_n=3_h=50_d=0.5__mvs=-1_mtiv=0_lr=0.0001_w2v=g	ACC	0.787800014	0
LSTM_n=5_h=300_d=0__mvs=-1_mtiv=0_lr=0.0001_w2v=g	ACC	0.787800014	0
RNN_n=2_h=300_d=0.9__mvs=-1_mtiv=0_lr=0.0001_w2v=g	ACC	0.787800014	0
RNN_n=5_h=50_d=0__mvs=-1_mtiv=0_lr=0.0001_w2v=g	ACC	0.787800014	0
RNN_n=2_h=50_d=0.9__mvs=-1_mtiv=0_lr=0.0001_w2v=g	ACC	0.7867000103	0
RNN_n=3_h=300_d=0.9__mvs=-1_mtiv=0_lr=0.0001_w2v=g	ACC	0.7843999863	0
RNN_n=3_h=50_d=0__mvs=-1_mtiv=0_lr=0.0001_w2v=g	ACC	0.7843999863	0
LSTM_n=5_h=50_d=0.9__mvs=-1_mtiv=0_lr=0.0001_w2v=g	ACC	0.7832999825	0
RNN_n=2_h=150_d=0.5__mvs=-1_mtiv=0_lr=0.0001_w2v=g	ACC	0.7832999825	0
RNN_n=3_h=150_d=0__mvs=-1_mtiv=0_lr=0.0001_w2v=g	ACC	0.7832999825	0
RNN_n=2_h=50_d=0__mvs=-1_mtiv=0_lr=0.0001_w2v=g	ACC	0.7821000218	0
RNN_n=3_h=300_d=0.5__mvs=-1_mtiv=0_lr=0.0001_w2v=g	ACC	0.7821000218	0
BASELINE__mvs=-1_mtiv=0_lr=0.0001_w2v=g	ACC	0.7816400051	0.004847935884
RNN_n=5_h=150_d=0__mvs=-1_mtiv=0_lr=0.0001_w2v=g	ACC	0.7810000181	0
RNN_n=5_h=300_d=0.5__mvs=-1_mtiv=0_lr=0.0001_w2v=g	ACC	0.7797999978	0
RNN_n=5_h=300_d=0__mvs=-1_mtiv=0_lr=0.0001_w2v=g	ACC	0.7786999941	0
RNN_n=2_h=300_d=0__mvs=-1_mtiv=0_lr=0.0001_w2v=g	ACC	0.7764000297	0
RNN_n=3_h=50_d=0.9__mvs=-1_mtiv=0_lr=0.0001_w2v=g	ACC	0.7741000056	0
RNN_n=2_h=150_d=0__mvs=-1_mtiv=0_lr=0.0001_w2v=g	ACC	0.7717999816	0
RNN_n=2_h=300_d=0.5__mvs=-1_mtiv=0_lr=0.0001_w2v=g	ACC	0.7695000172	0
RNN_n=3_h=300_d=0__mvs=-1_mtiv=0_lr=0.0001_w2v=g	ACC	0.7443000078	0
GRU_n=5_h=50_d=0.9__mvs=-1_mtiv=0_lr=0.0001_w2v=g	ACC	0.7317000031	0
RNN_n=5_h=150_d=0.9__mvs=-1_mtiv=0_lr=0.0001_w2v=g	ACC	0.5091999769	0
RNN_n=5_h=300_d=0.9__mvs=-1_mtiv=0_lr=0.0001_w2v=g	ACC	0.5091999769	0
RNN_n=5_h=50_d=0.9__mvs=-1_mtiv=0_lr=0.0001_w2v=g	ACC	0.5091999769	0
RNN_n=5_h=150_d=0.9__mvs=-1_mtiv=0_lr=0.0001_w2v=g	F1	nan	nan
RNN_n=5_h=300_d=0.9__mvs=-1_mtiv=0_lr=0.0001_w2v=g	F1	nan	nan
RNN_n=5_h=50_d=0.9__mvs=-1_mtiv=0_lr=0.0001_w2v=g	F1	nan	nan
LSTM_n=5_h=150_d=0.5__mvs=-1_mtiv=0_lr=0.0001_w2v=g	F1	0.8199999928	0
GRU_n=5_h=150_d=0.5__mvs=-1_mtiv=0_lr=0.0001_w2v=g	F1	0.8154000044	0
GRU_n=2_h=150_d=0.5__mvs=-1_mtiv=0_lr=0.0001_w2v=g	F1	0.814800024	0
LSTM_n=2_h=50_d=0.5__mvs=-1_mtiv=0_lr=0.0001_w2v=g	F1	0.8136000037	0
LSTM_n=5_h=300_d=0.5__mvs=-1_mtiv=0_lr=0.0001_w2v=g	F1	0.8130999804	0
LSTM_n=3_h=300_d=0__mvs=-1_mtiv=0_lr=0.0001_w2v=g	F1	0.8087000251	0
LSTM_n=2_h=50_d=0.9__mvs=-1_mtiv=0_lr=0.0001_w2v=g	F1	0.8083999753	0
LSTM_n=2_h=300_d=0__mvs=-1_mtiv=0_lr=0.0001_w2v=g	F1	0.804799974	0
GRU_n=2_h=150_d=0__mvs=-1_mtiv=0_lr=0.0001_w2v=g	F1	0.8037799954	0.007688264581
GRU_n=3_h=50_d=0.5__mvs=-1_mtiv=0_lr=0.0001_w2v=g	F1	0.803760004	0.006490180911
LSTM_n=3_h=150_d=0__mvs=-1_mtiv=0_lr=0.0001_w2v=g	F1	0.8033999801	0
LSTM_n=2_h=150_d=0.9__mvs=-1_mtiv=0_lr=0.0001_w2v=g	F1	0.8033000231	0
LSTM_n=2_h=300_d=0.9__mvs=-1_mtiv=0_lr=0.0001_w2v=g	F1	0.8023999929	0
LSTM_n=3_h=300_d=0.5__mvs=-1_mtiv=0_lr=0.0001_w2v=g	F1	0.8008999825	0
LSTM_n=3_h=150_d=0.5__mvs=-1_mtiv=0_lr=0.0001_w2v=g	F1	0.8004999757	0
RNN_n=5_h=50_d=0.5__mvs=-1_mtiv=0_lr=0.0001_w2v=g	F1	0.8004999757	0
LSTM_n=3_h=300_d=0.9__mvs=-1_mtiv=0_lr=0.0001_w2v=g	F1	0.8000000119	0
GRU_n=5_h=300_d=0__mvs=-1_mtiv=0_lr=0.0001_w2v=g	F1	0.7996399999	0.0111691699
LSTM_n=2_h=50_d=0__mvs=-1_mtiv=0_lr=0.0001_w2v=g	F1	0.7994999886	0
GRU_n=3_h=300_d=0__mvs=-1_mtiv=0_lr=0.0001_w2v=g	F1	0.7987999916	0.01571278295
GRU_n=3_h=150_d=0.9__mvs=-1_mtiv=0_lr=0.0001_w2v=g	F1	0.7982000113	0
GRU_n=5_h=300_d=0.9__mvs=-1_mtiv=0_lr=0.0001_w2v=g	F1	0.7982000113	0
LSTM_n=5_h=150_d=0.9__mvs=-1_mtiv=0_lr=0.0001_w2v=g	F1	0.7982000113	0
RNN_n=3_h=150_d=0.9__mvs=-1_mtiv=0_lr=0.0001_w2v=g	F1	0.7972999811	0
LSTM_n=5_h=300_d=0.9__mvs=-1_mtiv=0_lr=0.0001_w2v=g	F1	0.7972000241	0
GRU_n=2_h=50_d=0.5__mvs=-1_mtiv=0_lr=0.0001_w2v=g	F1	0.7971000075	0.005696313711
GRU_n=3_h=50_d=0__mvs=-1_mtiv=0_lr=0.0001_w2v=g	F1	0.7970200062	0.006884587257
LSTM_n=5_h=50_d=0__mvs=-1_mtiv=0_lr=0.0001_w2v=g	F1	0.7965999842	0
GRU_n=5_h=300_d=0.5__mvs=-1_mtiv=0_lr=0.0001_w2v=g	F1	0.7965000272	0
GRU_n=5_h=50_d=0.5__mvs=-1_mtiv=0_lr=0.0001_w2v=g	F1	0.796299994	0
GRU_n=3_h=300_d=0.5__mvs=-1_mtiv=0_lr=0.0001_w2v=g	F1	0.7961599946	0.007772930297
GRU_n=5_h=50_d=0__mvs=-1_mtiv=0_lr=0.0001_w2v=g	F1	0.7959599972	0.007158389354
GRU_n=5_h=150_d=0.9__mvs=-1_mtiv=0_lr=0.0001_w2v=g	F1	0.7951999903	0
GRU_n=2_h=300_d=0.5__mvs=-1_mtiv=0_lr=0.0001_w2v=g	F1	0.7951799989	0.01333183483
LSTM_n=2_h=150_d=0__mvs=-1_mtiv=0_lr=0.0001_w2v=g	F1	0.7947000265	0
LSTM_n=3_h=50_d=0.9__mvs=-1_mtiv=0_lr=0.0001_w2v=g	F1	0.7943000197	0
LSTM_n=5_h=50_d=0.5__mvs=-1_mtiv=0_lr=0.0001_w2v=g	F1	0.7943000197	0
LSTM_n=3_h=150_d=0.9__mvs=-1_mtiv=0_lr=0.0001_w2v=g	F1	0.7942000031	0
GRU_n=2_h=300_d=0__mvs=-1_mtiv=0_lr=0.0001_w2v=g	F1	0.7935199976	0.01480882369
GRU_n=2_h=150_d=0.9__mvs=-1_mtiv=0_lr=0.0001_w2v=g	F1	0.7933999896	0
LSTM_n=3_h=50_d=0__mvs=-1_mtiv=0_lr=0.0001_w2v=g	F1	0.793299973	0
GRU_n=2_h=50_d=0.9__mvs=-1_mtiv=0_lr=0.0001_w2v=g	F1	0.7932400107	0.006613504728
LSTM_n=5_h=300_d=0__mvs=-1_mtiv=0_lr=0.0001_w2v=g	F1	0.7928000093	0
LSTM_n=5_h=150_d=0__mvs=-1_mtiv=0_lr=0.0001_w2v=g	F1	0.7921000123	0
GRU_n=2_h=300_d=0.9__mvs=-1_mtiv=0_lr=0.0001_w2v=g	F1	0.791900003	0.008617418536
LSTM_n=2_h=150_d=0.5__mvs=-1_mtiv=0_lr=0.0001_w2v=g	F1	0.7910000086	0
RNN_n=2_h=300_d=0.9__mvs=-1_mtiv=0_lr=0.0001_w2v=g	F1	0.7910000086	0
GRU_n=3_h=300_d=0.9__mvs=-1_mtiv=0_lr=0.0001_w2v=g	F1	0.7902400136	0.01469115599
RNN_n=2_h=50_d=0.5__mvs=-1_mtiv=0_lr=0.0001_w2v=g	F1	0.7900000215	0
RNN_n=3_h=150_d=0.5__mvs=-1_mtiv=0_lr=0.0001_w2v=g	F1	0.7896000147	0
GRU_n=3_h=150_d=0.5__mvs=-1_mtiv=0_lr=0.0001_w2v=g	F1	0.7886999846	0
RNN_n=2_h=150_d=0.9__mvs=-1_mtiv=0_lr=0.0001_w2v=g	F1	0.7879999876	0
RNN_n=3_h=300_d=0.5__mvs=-1_mtiv=0_lr=0.0001_w2v=g	F1	0.787899971	0
RNN_n=3_h=50_d=0.9__mvs=-1_mtiv=0_lr=0.0001_w2v=g	F1	0.7875000238	0
RNN_n=2_h=50_d=0.9__mvs=-1_mtiv=0_lr=0.0001_w2v=g	F1	0.7871999741	0
RNN_n=3_h=50_d=0__mvs=-1_mtiv=0_lr=0.0001_w2v=g	F1	0.7868000269	0
GRU_n=3_h=50_d=0.9__mvs=-1_mtiv=0_lr=0.0001_w2v=g	F1	0.7863600016	0.006006852079
RNN_n=2_h=150_d=0.5__mvs=-1_mtiv=0_lr=0.0001_w2v=g	F1	0.7860000134	0
GRU_n=2_h=50_d=0__mvs=-1_mtiv=0_lr=0.0001_w2v=g	F1	0.7859499902	0.009295815311
RNN_n=3_h=50_d=0.5__mvs=-1_mtiv=0_lr=0.0001_w2v=g	F1	0.7786999941	0
RNN_n=5_h=50_d=0__mvs=-1_mtiv=0_lr=0.0001_w2v=g	F1	0.7778999805	0
LSTM_n=5_h=50_d=0.9__mvs=-1_mtiv=0_lr=0.0001_w2v=g	F1	0.7763000131	0
RNN_n=2_h=50_d=0__mvs=-1_mtiv=0_lr=0.0001_w2v=g	F1	0.7759000063	0
RNN_n=5_h=150_d=0.5__mvs=-1_mtiv=0_lr=0.0001_w2v=g	F1	0.7754999995	0
LSTM_n=2_h=300_d=0.5__mvs=-1_mtiv=0_lr=0.0001_w2v=g	F1	0.7753999829	0
LSTM_n=3_h=50_d=0.5__mvs=-1_mtiv=0_lr=0.0001_w2v=g	F1	0.7752000093	0
RNN_n=3_h=150_d=0__mvs=-1_mtiv=0_lr=0.0001_w2v=g	F1	0.7731000185	0
RNN_n=2_h=300_d=0__mvs=-1_mtiv=0_lr=0.0001_w2v=g	F1	0.7730000019	0
GRU_n=5_h=150_d=0__mvs=-1_mtiv=0_lr=0.0001_w2v=g	F1	0.7724000216	0
BASELINE__mvs=-1_mtiv=0_lr=0.0001_w2v=g	F1	0.7717200041	0.006941303626
RNN_n=5_h=300_d=0.5__mvs=-1_mtiv=0_lr=0.0001_w2v=g	F1	0.7698000073	0
RNN_n=2_h=150_d=0__mvs=-1_mtiv=0_lr=0.0001_w2v=g	F1	0.7688999772	0
RNN_n=3_h=300_d=0.9__mvs=-1_mtiv=0_lr=0.0001_w2v=g	F1	0.7656000257	0
RNN_n=5_h=300_d=0__mvs=-1_mtiv=0_lr=0.0001_w2v=g	F1	0.7613999844	0
GRU_n=3_h=150_d=0__mvs=-1_mtiv=0_lr=0.0001_w2v=g	F1	0.760800004	0
RNN_n=5_h=150_d=0__mvs=-1_mtiv=0_lr=0.0001_w2v=g	F1	0.7603999972	0
RNN_n=2_h=300_d=0.5__mvs=-1_mtiv=0_lr=0.0001_w2v=g	F1	0.7540000081	0
GRU_n=5_h=50_d=0.9__mvs=-1_mtiv=0_lr=0.0001_w2v=g	F1	0.7322999835	0
RNN_n=3_h=300_d=0__mvs=-1_mtiv=0_lr=0.0001_w2v=g	F1	0.6958000064	0
RNN_n=5_h=50_d=0.9__mvs=-1_mtiv=0_lr=0.0001_w2v=g	LOSS	0.693599999	0
RNN_n=5_h=150_d=0.9__mvs=-1_mtiv=0_lr=0.0001_w2v=g	LOSS	0.6930000186	0
RNN_n=5_h=300_d=0.9__mvs=-1_mtiv=0_lr=0.0001_w2v=g	LOSS	0.6923999786	0
GRU_n=5_h=50_d=0.9__mvs=-1_mtiv=0_lr=0.0001_w2v=g	LOSS	0.5605000257	0
RNN_n=3_h=50_d=0.9__mvs=-1_mtiv=0_lr=0.0001_w2v=g	LOSS	0.5473999977	0
RNN_n=2_h=50_d=0.9__mvs=-1_mtiv=0_lr=0.0001_w2v=g	LOSS	0.5142999887	0
RNN_n=3_h=150_d=0.9__mvs=-1_mtiv=0_lr=0.0001_w2v=g	LOSS	0.5124999881	0
GRU_n=5_h=150_d=0.9__mvs=-1_mtiv=0_lr=0.0001_w2v=g	LOSS	0.5123000145	0
LSTM_n=5_h=50_d=0.9__mvs=-1_mtiv=0_lr=0.0001_w2v=g	LOSS	0.507700026	0
GRU_n=3_h=50_d=0.9__mvs=-1_mtiv=0_lr=0.0001_w2v=g	LOSS	0.5016200006	0.008745600493
RNN_n=2_h=150_d=0.5__mvs=-1_mtiv=0_lr=0.0001_w2v=g	LOSS	0.491899997	0
GRU_n=5_h=300_d=0.9__mvs=-1_mtiv=0_lr=0.0001_w2v=g	LOSS	0.4873999953	0
GRU_n=3_h=150_d=0.9__mvs=-1_mtiv=0_lr=0.0001_w2v=g	LOSS	0.4871999919	0
RNN_n=3_h=300_d=0.9__mvs=-1_mtiv=0_lr=0.0001_w2v=g	LOSS	0.4871000051	0
LSTM_n=2_h=50_d=0.5__mvs=-1_mtiv=0_lr=0.0001_w2v=g	LOSS	0.4821999967	0
RNN_n=3_h=300_d=0.5__mvs=-1_mtiv=0_lr=0.0001_w2v=g	LOSS	0.4817999899	0
RNN_n=2_h=150_d=0__mvs=-1_mtiv=0_lr=0.0001_w2v=g	LOSS	0.4814999998	0
RNN_n=3_h=300_d=0__mvs=-1_mtiv=0_lr=0.0001_w2v=g	LOSS	0.4771000147	0
RNN_n=2_h=50_d=0.5__mvs=-1_mtiv=0_lr=0.0001_w2v=g	LOSS	0.4740000069	0
GRU_n=2_h=50_d=0.9__mvs=-1_mtiv=0_lr=0.0001_w2v=g	LOSS	0.4735599995	0.007112404702
RNN_n=3_h=150_d=0__mvs=-1_mtiv=0_lr=0.0001_w2v=g	LOSS	0.4733999968	0
LSTM_n=3_h=50_d=0.9__mvs=-1_mtiv=0_lr=0.0001_w2v=g	LOSS	0.4729000032	0
LSTM_n=5_h=50_d=0.5__mvs=-1_mtiv=0_lr=0.0001_w2v=g	LOSS	0.4727999866	0
LSTM_n=5_h=150_d=0.9__mvs=-1_mtiv=0_lr=0.0001_w2v=g	LOSS	0.472600013	0
RNN_n=5_h=300_d=0__mvs=-1_mtiv=0_lr=0.0001_w2v=g	LOSS	0.4724000096	0
RNN_n=2_h=150_d=0.9__mvs=-1_mtiv=0_lr=0.0001_w2v=g	LOSS	0.4720999897	0
BASELINE__mvs=-1_mtiv=0_lr=0.0001_w2v=g	LOSS	0.4719600022	0.002835210228
RNN_n=5_h=300_d=0.5__mvs=-1_mtiv=0_lr=0.0001_w2v=g	LOSS	0.4715999961	0
LSTM_n=5_h=300_d=0.9__mvs=-1_mtiv=0_lr=0.0001_w2v=g	LOSS	0.4704999924	0
RNN_n=3_h=50_d=0.5__mvs=-1_mtiv=0_lr=0.0001_w2v=g	LOSS	0.4688999951	0
RNN_n=2_h=300_d=0.5__mvs=-1_mtiv=0_lr=0.0001_w2v=g	LOSS	0.4686000049	0
RNN_n=2_h=300_d=0__mvs=-1_mtiv=0_lr=0.0001_w2v=g	LOSS	0.4684000015	0
RNN_n=2_h=300_d=0.9__mvs=-1_mtiv=0_lr=0.0001_w2v=g	LOSS	0.4679999948	0
RNN_n=5_h=50_d=0__mvs=-1_mtiv=0_lr=0.0001_w2v=g	LOSS	0.467599988	0
RNN_n=5_h=50_d=0.5__mvs=-1_mtiv=0_lr=0.0001_w2v=g	LOSS	0.4674000144	0
RNN_n=3_h=50_d=0__mvs=-1_mtiv=0_lr=0.0001_w2v=g	LOSS	0.467200011	0
RNN_n=2_h=50_d=0__mvs=-1_mtiv=0_lr=0.0001_w2v=g	LOSS	0.4641000032	0
LSTM_n=2_h=50_d=0.9__mvs=-1_mtiv=0_lr=0.0001_w2v=g	LOSS	0.4602000117	0
RNN_n=5_h=150_d=0__mvs=-1_mtiv=0_lr=0.0001_w2v=g	LOSS	0.4582999945	0
LSTM_n=5_h=150_d=0.5__mvs=-1_mtiv=0_lr=0.0001_w2v=g	LOSS	0.4578999877	0
RNN_n=5_h=150_d=0.5__mvs=-1_mtiv=0_lr=0.0001_w2v=g	LOSS	0.4564999938	0
LSTM_n=3_h=300_d=0.5__mvs=-1_mtiv=0_lr=0.0001_w2v=g	LOSS	0.4557999969	0
LSTM_n=2_h=50_d=0__mvs=-1_mtiv=0_lr=0.0001_w2v=g	LOSS	0.4548999965	0
RNN_n=3_h=150_d=0.5__mvs=-1_mtiv=0_lr=0.0001_w2v=g	LOSS	0.4535000026	0
GRU_n=2_h=50_d=0.5__mvs=-1_mtiv=0_lr=0.0001_w2v=g	LOSS	0.4483999968	0.00731901611
LSTM_n=5_h=50_d=0__mvs=-1_mtiv=0_lr=0.0001_w2v=g	LOSS	0.4478000104	0
GRU_n=3_h=50_d=0.5__mvs=-1_mtiv=0_lr=0.0001_w2v=g	LOSS	0.4477199972	0.0066930976
LSTM_n=3_h=50_d=0.5__mvs=-1_mtiv=0_lr=0.0001_w2v=g	LOSS	0.4474000037	0
LSTM_n=3_h=150_d=0.9__mvs=-1_mtiv=0_lr=0.0001_w2v=g	LOSS	0.4471000135	0
GRU_n=5_h=50_d=0.5__mvs=-1_mtiv=0_lr=0.0001_w2v=g	LOSS	0.4465999901	0
LSTM_n=5_h=300_d=0__mvs=-1_mtiv=0_lr=0.0001_w2v=g	LOSS	0.4465000033	0
GRU_n=5_h=50_d=0__mvs=-1_mtiv=0_lr=0.0001_w2v=g	LOSS	0.4451799929	0.005698917695
GRU_n=2_h=50_d=0__mvs=-1_mtiv=0_lr=0.0001_w2v=g	LOSS	0.4443750083	0.005888286921
LSTM_n=2_h=150_d=0.9__mvs=-1_mtiv=0_lr=0.0001_w2v=g	LOSS	0.4438000023	0
GRU_n=5_h=150_d=0.5__mvs=-1_mtiv=0_lr=0.0001_w2v=g	LOSS	0.4429999888	0
GRU_n=3_h=300_d=0.9__mvs=-1_mtiv=0_lr=0.0001_w2v=g	LOSS	0.4427200019	0.01402774568
GRU_n=3_h=50_d=0__mvs=-1_mtiv=0_lr=0.0001_w2v=g	LOSS	0.4426200032	0.003364158633
LSTM_n=2_h=150_d=0__mvs=-1_mtiv=0_lr=0.0001_w2v=g	LOSS	0.4424999952	0
GRU_n=2_h=150_d=0.9__mvs=-1_mtiv=0_lr=0.0001_w2v=g	LOSS	0.4420999885	0
LSTM_n=3_h=150_d=0__mvs=-1_mtiv=0_lr=0.0001_w2v=g	LOSS	0.4420000017	0
LSTM_n=3_h=50_d=0__mvs=-1_mtiv=0_lr=0.0001_w2v=g	LOSS	0.4408999979	0
LSTM_n=3_h=300_d=0__mvs=-1_mtiv=0_lr=0.0001_w2v=g	LOSS	0.4404999912	0
GRU_n=2_h=150_d=0.5__mvs=-1_mtiv=0_lr=0.0001_w2v=g	LOSS	0.4388000071	0
GRU_n=3_h=150_d=0__mvs=-1_mtiv=0_lr=0.0001_w2v=g	LOSS	0.4368999898	0
LSTM_n=2_h=150_d=0.5__mvs=-1_mtiv=0_lr=0.0001_w2v=g	LOSS	0.4357999861	0
LSTM_n=3_h=300_d=0.9__mvs=-1_mtiv=0_lr=0.0001_w2v=g	LOSS	0.4352000058	0
LSTM_n=3_h=150_d=0.5__mvs=-1_mtiv=0_lr=0.0001_w2v=g	LOSS	0.4348999858	0
GRU_n=3_h=150_d=0.5__mvs=-1_mtiv=0_lr=0.0001_w2v=g	LOSS	0.4345999956	0
GRU_n=3_h=300_d=0.5__mvs=-1_mtiv=0_lr=0.0001_w2v=g	LOSS	0.4341199994	0.01076742857
GRU_n=2_h=150_d=0__mvs=-1_mtiv=0_lr=0.0001_w2v=g	LOSS	0.4333599985	0.00627426087
LSTM_n=5_h=150_d=0__mvs=-1_mtiv=0_lr=0.0001_w2v=g	LOSS	0.4330999851	0
GRU_n=2_h=300_d=0.5__mvs=-1_mtiv=0_lr=0.0001_w2v=g	LOSS	0.4327199996	0.00457794896
GRU_n=3_h=300_d=0__mvs=-1_mtiv=0_lr=0.0001_w2v=g	LOSS	0.4315999985	0.017470091
LSTM_n=2_h=300_d=0.9__mvs=-1_mtiv=0_lr=0.0001_w2v=g	LOSS	0.4302000105	0
LSTM_n=2_h=300_d=0__mvs=-1_mtiv=0_lr=0.0001_w2v=g	LOSS	0.4300999939	0
GRU_n=2_h=300_d=0__mvs=-1_mtiv=0_lr=0.0001_w2v=g	LOSS	0.4298200071	0.005873807403
LSTM_n=2_h=300_d=0.5__mvs=-1_mtiv=0_lr=0.0001_w2v=g	LOSS	0.4298000038	0
GRU_n=5_h=300_d=0.5__mvs=-1_mtiv=0_lr=0.0001_w2v=g	LOSS	0.4271000028	0
GRU_n=2_h=300_d=0.9__mvs=-1_mtiv=0_lr=0.0001_w2v=g	LOSS	0.4263999999	0.008562946649
LSTM_n=5_h=300_d=0.5__mvs=-1_mtiv=0_lr=0.0001_w2v=g	LOSS	0.4257000089	0
GRU_n=5_h=300_d=0__mvs=-1_mtiv=0_lr=0.0001_w2v=g	LOSS	0.4228799999	0.003071417526
GRU_n=5_h=150_d=0__mvs=-1_mtiv=0_lr=0.0001_w2v=g	LOSS	0.4210999906	0
RNN_n=5_h=150_d=0.9__mvs=-1_mtiv=0_lr=0.0001_w2v=g	PRE	nan	nan
RNN_n=5_h=300_d=0.9__mvs=-1_mtiv=0_lr=0.0001_w2v=g	PRE	nan	nan
RNN_n=5_h=50_d=0.9__mvs=-1_mtiv=0_lr=0.0001_w2v=g	PRE	nan	nan
GRU_n=3_h=150_d=0__mvs=-1_mtiv=0_lr=0.0001_w2v=g	PRE	0.863499999	0
GRU_n=5_h=150_d=0__mvs=-1_mtiv=0_lr=0.0001_w2v=g	PRE	0.8531000018	0
LSTM_n=5_h=300_d=0.5__mvs=-1_mtiv=0_lr=0.0001_w2v=g	PRE	0.8460000157	0
GRU_n=5_h=300_d=0.5__mvs=-1_mtiv=0_lr=0.0001_w2v=g	PRE	0.8432999849	0
LSTM_n=5_h=150_d=0__mvs=-1_mtiv=0_lr=0.0001_w2v=g	PRE	0.8421000242	0
GRU_n=5_h=300_d=0__mvs=-1_mtiv=0_lr=0.0001_w2v=g	PRE	0.8413599968	0.01481603023
GRU_n=2_h=300_d=0.9__mvs=-1_mtiv=0_lr=0.0001_w2v=g	PRE	0.8403000116	0.008599767394
GRU_n=3_h=300_d=0__mvs=-1_mtiv=0_lr=0.0001_w2v=g	PRE	0.8376600027	0.02621920683
LSTM_n=2_h=300_d=0.5__mvs=-1_mtiv=0_lr=0.0001_w2v=g	PRE	0.8374000192	0
RNN_n=3_h=300_d=0__mvs=-1_mtiv=0_lr=0.0001_w2v=g	PRE	0.8360999823	0
LSTM_n=3_h=150_d=0.5__mvs=-1_mtiv=0_lr=0.0001_w2v=g	PRE	0.8349999785	0
GRU_n=3_h=300_d=0.5__mvs=-1_mtiv=0_lr=0.0001_w2v=g	PRE	0.8336600065	0.01109768242
GRU_n=3_h=150_d=0.5__mvs=-1_mtiv=0_lr=0.0001_w2v=g	PRE	0.8316000104	0
GRU_n=2_h=300_d=0.5__mvs=-1_mtiv=0_lr=0.0001_w2v=g	PRE	0.8303200126	0.01547634801
GRU_n=2_h=300_d=0__mvs=-1_mtiv=0_lr=0.0001_w2v=g	PRE	0.8290399909	0.01421627438
LSTM_n=2_h=300_d=0__mvs=-1_mtiv=0_lr=0.0001_w2v=g	PRE	0.825600028	0
LSTM_n=3_h=150_d=0__mvs=-1_mtiv=0_lr=0.0001_w2v=g	PRE	0.8251000047	0
LSTM_n=5_h=50_d=0__mvs=-1_mtiv=0_lr=0.0001_w2v=g	PRE	0.8213000298	0
RNN_n=5_h=150_d=0__mvs=-1_mtiv=0_lr=0.0001_w2v=g	PRE	0.8210999966	0
RNN_n=3_h=300_d=0.9__mvs=-1_mtiv=0_lr=0.0001_w2v=g	PRE	0.820900023	0
LSTM_n=3_h=300_d=0.9__mvs=-1_mtiv=0_lr=0.0001_w2v=g	PRE	0.8205999732	0
LSTM_n=3_h=150_d=0.9__mvs=-1_mtiv=0_lr=0.0001_w2v=g	PRE	0.8188999891	0
GRU_n=3_h=300_d=0.9__mvs=-1_mtiv=0_lr=0.0001_w2v=g	PRE	0.8182400107	0.0156028873
GRU_n=5_h=50_d=0__mvs=-1_mtiv=0_lr=0.0001_w2v=g	PRE	0.8175199986	0.01281270144
LSTM_n=2_h=150_d=0__mvs=-1_mtiv=0_lr=0.0001_w2v=g	PRE	0.8173000216	0
LSTM_n=3_h=50_d=0__mvs=-1_mtiv=0_lr=0.0001_w2v=g	PRE	0.8167999983	0
RNN_n=5_h=150_d=0.5__mvs=-1_mtiv=0_lr=0.0001_w2v=g	PRE	0.8165000081	0
GRU_n=3_h=50_d=0__mvs=-1_mtiv=0_lr=0.0001_w2v=g	PRE	0.8161800027	0.01194761422
GRU_n=5_h=150_d=0.5__mvs=-1_mtiv=0_lr=0.0001_w2v=g	PRE	0.8154000044	0
LSTM_n=2_h=150_d=0.9__mvs=-1_mtiv=0_lr=0.0001_w2v=g	PRE	0.814899981	0
RNN_n=2_h=150_d=0.9__mvs=-1_mtiv=0_lr=0.0001_w2v=g	PRE	0.8133999705	0
GRU_n=2_h=50_d=0__mvs=-1_mtiv=0_lr=0.0001_w2v=g	PRE	0.8108250052	0.01120990966
GRU_n=5_h=150_d=0.9__mvs=-1_mtiv=0_lr=0.0001_w2v=g	PRE	0.8106999993	0
LSTM_n=2_h=50_d=0.9__mvs=-1_mtiv=0_lr=0.0001_w2v=g	PRE	0.8083999753	0
RNN_n=5_h=300_d=0__mvs=-1_mtiv=0_lr=0.0001_w2v=g	PRE	0.8083999753	0
LSTM_n=2_h=300_d=0.9__mvs=-1_mtiv=0_lr=0.0001_w2v=g	PRE	0.8080999851	0
LSTM_n=3_h=50_d=0.5__mvs=-1_mtiv=0_lr=0.0001_w2v=g	PRE	0.8076000214	0
GRU_n=2_h=150_d=0.5__mvs=-1_mtiv=0_lr=0.0001_w2v=g	PRE	0.8072999716	0
RNN_n=3_h=50_d=0.5__mvs=-1_mtiv=0_lr=0.0001_w2v=g	PRE	0.8069999814	0
GRU_n=2_h=150_d=0__mvs=-1_mtiv=0_lr=0.0001_w2v=g	PRE	0.8044400096	0.02085527093
LSTM_n=2_h=150_d=0.5__mvs=-1_mtiv=0_lr=0.0001_w2v=g	PRE	0.8043000102	0
LSTM_n=3_h=50_d=0.9__mvs=-1_mtiv=0_lr=0.0001_w2v=g	PRE	0.8037999868	0
LSTM_n=5_h=50_d=0.5__mvs=-1_mtiv=0_lr=0.0001_w2v=g	PRE	0.8037999868	0
GRU_n=2_h=150_d=0.9__mvs=-1_mtiv=0_lr=0.0001_w2v=g	PRE	0.8019000292	0
RNN_n=5_h=50_d=0__mvs=-1_mtiv=0_lr=0.0001_w2v=g	PRE	0.8000000119	0
GRU_n=2_h=50_d=0.5__mvs=-1_mtiv=0_lr=0.0001_w2v=g	PRE	0.7994800091	0.01087003263
RNN_n=3_h=150_d=0.5__mvs=-1_mtiv=0_lr=0.0001_w2v=g	PRE	0.7990000248	0
LSTM_n=5_h=300_d=0.9__mvs=-1_mtiv=0_lr=0.0001_w2v=g	PRE	0.7953000069	0
RNN_n=3_h=150_d=0__mvs=-1_mtiv=0_lr=0.0001_w2v=g	PRE	0.7950999737	0
GRU_n=5_h=50_d=0.5__mvs=-1_mtiv=0_lr=0.0001_w2v=g	PRE	0.7935000062	0
LSTM_n=3_h=300_d=0__mvs=-1_mtiv=0_lr=0.0001_w2v=g	PRE	0.793299973	0
BASELINE__mvs=-1_mtiv=0_lr=0.0001_w2v=g	PRE	0.7925599933	0.008798772566
RNN_n=2_h=300_d=0.5__mvs=-1_mtiv=0_lr=0.0001_w2v=g	PRE	0.7918000221	0
GRU_n=3_h=50_d=0.9__mvs=-1_mtiv=0_lr=0.0001_w2v=g	PRE	0.7911199927	0.008064089812
GRU_n=2_h=50_d=0.9__mvs=-1_mtiv=0_lr=0.0001_w2v=g	PRE	0.7909999967	0.02492556771
RNN_n=5_h=300_d=0.5__mvs=-1_mtiv=0_lr=0.0001_w2v=g	PRE	0.7906000018	0
GRU_n=3_h=50_d=0.5__mvs=-1_mtiv=0_lr=0.0001_w2v=g	PRE	0.7896000028	0.002342648038
LSTM_n=3_h=300_d=0.5__mvs=-1_mtiv=0_lr=0.0001_w2v=g	PRE	0.7890999913	0
LSTM_n=2_h=50_d=0__mvs=-1_mtiv=0_lr=0.0001_w2v=g	PRE	0.7886000276	0
LSTM_n=2_h=50_d=0.5__mvs=-1_mtiv=0_lr=0.0001_w2v=g	PRE	0.7876999974	0
LSTM_n=5_h=50_d=0.9__mvs=-1_mtiv=0_lr=0.0001_w2v=g	PRE	0.7865999937	0
RNN_n=2_h=50_d=0__mvs=-1_mtiv=0_lr=0.0001_w2v=g	PRE	0.7832999825	0
LSTM_n=5_h=150_d=0.5__mvs=-1_mtiv=0_lr=0.0001_w2v=g	PRE	0.7817999721	0
RNN_n=5_h=50_d=0.5__mvs=-1_mtiv=0_lr=0.0001_w2v=g	PRE	0.7817000151	0
RNN_n=3_h=150_d=0.9__mvs=-1_mtiv=0_lr=0.0001_w2v=g	PRE	0.7735999823	0
RNN_n=2_h=50_d=0.5__mvs=-1_mtiv=0_lr=0.0001_w2v=g	PRE	0.772300005	0
RNN_n=2_h=50_d=0.9__mvs=-1_mtiv=0_lr=0.0001_w2v=g	PRE	0.7713000178	0
LSTM_n=5_h=150_d=0.9__mvs=-1_mtiv=0_lr=0.0001_w2v=g	PRE	0.7712000012	0
RNN_n=2_h=300_d=0__mvs=-1_mtiv=0_lr=0.0001_w2v=g	PRE	0.7702999711	0
GRU_n=3_h=150_d=0.9__mvs=-1_mtiv=0_lr=0.0001_w2v=g	PRE	0.7671999931	0
GRU_n=5_h=300_d=0.9__mvs=-1_mtiv=0_lr=0.0001_w2v=g	PRE	0.7671999931	0
RNN_n=2_h=300_d=0.9__mvs=-1_mtiv=0_lr=0.0001_w2v=g	PRE	0.7659000158	0
RNN_n=2_h=150_d=0__mvs=-1_mtiv=0_lr=0.0001_w2v=g	PRE	0.7644000053	0
RNN_n=3_h=50_d=0__mvs=-1_mtiv=0_lr=0.0001_w2v=g	PRE	0.7642999887	0
RNN_n=2_h=150_d=0.5__mvs=-1_mtiv=0_lr=0.0001_w2v=g	PRE	0.7626000047	0
LSTM_n=5_h=300_d=0__mvs=-1_mtiv=0_lr=0.0001_w2v=g	PRE	0.7613000274	0
RNN_n=3_h=300_d=0.5__mvs=-1_mtiv=0_lr=0.0001_w2v=g	PRE	0.7542999983	0
RNN_n=3_h=50_d=0.9__mvs=-1_mtiv=0_lr=0.0001_w2v=g	PRE	0.7315000296	0
GRU_n=5_h=50_d=0.9__mvs=-1_mtiv=0_lr=0.0001_w2v=g	PRE	0.7174999714	0
LSTM_n=5_h=150_d=0.5__mvs=-1_mtiv=0_lr=0.0001_w2v=g	REC	0.8621000051	0
RNN_n=3_h=50_d=0.9__mvs=-1_mtiv=0_lr=0.0001_w2v=g	REC	0.8528000116	0
LSTM_n=2_h=50_d=0.5__mvs=-1_mtiv=0_lr=0.0001_w2v=g	REC	0.8410999775	0
GRU_n=3_h=150_d=0.9__mvs=-1_mtiv=0_lr=0.0001_w2v=g	REC	0.831799984	0
GRU_n=5_h=300_d=0.9__mvs=-1_mtiv=0_lr=0.0001_w2v=g	REC	0.831799984	0
LSTM_n=5_h=150_d=0.9__mvs=-1_mtiv=0_lr=0.0001_w2v=g	REC	0.8270999789	0
LSTM_n=5_h=300_d=0__mvs=-1_mtiv=0_lr=0.0001_w2v=g	REC	0.8270999789	0
LSTM_n=3_h=300_d=0__mvs=-1_mtiv=0_lr=0.0001_w2v=g	REC	0.8248000145	0
RNN_n=3_h=300_d=0.5__mvs=-1_mtiv=0_lr=0.0001_w2v=g	REC	0.8248000145	0
GRU_n=2_h=150_d=0.5__mvs=-1_mtiv=0_lr=0.0001_w2v=g	REC	0.8223999739	0
RNN_n=3_h=150_d=0.9__mvs=-1_mtiv=0_lr=0.0001_w2v=g	REC	0.8223999739	0
RNN_n=5_h=50_d=0.5__mvs=-1_mtiv=0_lr=0.0001_w2v=g	REC	0.8201000094	0
GRU_n=3_h=50_d=0.5__mvs=-1_mtiv=0_lr=0.0001_w2v=g	REC	0.8187000036	0.01545950312
RNN_n=2_h=300_d=0.9__mvs=-1_mtiv=0_lr=0.0001_w2v=g	REC	0.8177999854	0
GRU_n=5_h=150_d=0.5__mvs=-1_mtiv=0_lr=0.0001_w2v=g	REC	0.8154000044	0
LSTM_n=3_h=300_d=0.5__mvs=-1_mtiv=0_lr=0.0001_w2v=g	REC	0.8130999804	0
LSTM_n=2_h=50_d=0__mvs=-1_mtiv=0_lr=0.0001_w2v=g	REC	0.8106999993	0
RNN_n=2_h=150_d=0.5__mvs=-1_mtiv=0_lr=0.0001_w2v=g	REC	0.8106999993	0
RNN_n=3_h=50_d=0__mvs=-1_mtiv=0_lr=0.0001_w2v=g	REC	0.8106999993	0
LSTM_n=2_h=50_d=0.9__mvs=-1_mtiv=0_lr=0.0001_w2v=g	REC	0.8083999753	0
RNN_n=2_h=50_d=0.5__mvs=-1_mtiv=0_lr=0.0001_w2v=g	REC	0.8083999753	0
GRU_n=2_h=150_d=0__mvs=-1_mtiv=0_lr=0.0001_w2v=g	REC	0.8042199969	0.02397936616
RNN_n=2_h=50_d=0.9__mvs=-1_mtiv=0_lr=0.0001_w2v=g	REC	0.8036999702	0
GRU_n=5_h=50_d=0.5__mvs=-1_mtiv=0_lr=0.0001_w2v=g	REC	0.7990999818	0
LSTM_n=5_h=300_d=0.9__mvs=-1_mtiv=0_lr=0.0001_w2v=g	REC	0.7990999818	0
GRU_n=2_h=50_d=0.9__mvs=-1_mtiv=0_lr=0.0001_w2v=g	REC	0.797179985	0.02868039746
LSTM_n=2_h=300_d=0.9__mvs=-1_mtiv=0_lr=0.0001_w2v=g	REC	0.7967000008	0
GRU_n=2_h=50_d=0.5__mvs=-1_mtiv=0_lr=0.0001_w2v=g	REC	0.7953399897	0.0210174831
LSTM_n=2_h=150_d=0.9__mvs=-1_mtiv=0_lr=0.0001_w2v=g	REC	0.7921000123	0
GRU_n=2_h=150_d=0.9__mvs=-1_mtiv=0_lr=0.0001_w2v=g	REC	0.7850000262	0
LSTM_n=2_h=300_d=0__mvs=-1_mtiv=0_lr=0.0001_w2v=g	REC	0.7850000262	0
LSTM_n=3_h=50_d=0.9__mvs=-1_mtiv=0_lr=0.0001_w2v=g	REC	0.7850000262	0
LSTM_n=5_h=50_d=0.5__mvs=-1_mtiv=0_lr=0.0001_w2v=g	REC	0.7850000262	0
LSTM_n=3_h=150_d=0__mvs=-1_mtiv=0_lr=0.0001_w2v=g	REC	0.7827000022	0
LSTM_n=5_h=300_d=0.5__mvs=-1_mtiv=0_lr=0.0001_w2v=g	REC	0.7827000022	0
GRU_n=3_h=50_d=0.9__mvs=-1_mtiv=0_lr=0.0001_w2v=g	REC	0.7817799926	0.01041082524
GRU_n=5_h=150_d=0.9__mvs=-1_mtiv=0_lr=0.0001_w2v=g	REC	0.7803999782	0
LSTM_n=3_h=300_d=0.9__mvs=-1_mtiv=0_lr=0.0001_w2v=g	REC	0.7803999782	0
RNN_n=3_h=150_d=0.5__mvs=-1_mtiv=0_lr=0.0001_w2v=g	REC	0.7803999782	0
GRU_n=3_h=50_d=0__mvs=-1_mtiv=0_lr=0.0001_w2v=g	REC	0.7789600015	0.01111693828
LSTM_n=2_h=150_d=0.5__mvs=-1_mtiv=0_lr=0.0001_w2v=g	REC	0.7779999971	0
GRU_n=5_h=50_d=0__mvs=-1_mtiv=0_lr=0.0001_w2v=g	REC	0.7761599898	0.02081043779
RNN_n=2_h=300_d=0__mvs=-1_mtiv=0_lr=0.0001_w2v=g	REC	0.7756999731	0
LSTM_n=2_h=150_d=0__mvs=-1_mtiv=0_lr=0.0001_w2v=g	REC	0.7734000087	0
LSTM_n=5_h=50_d=0__mvs=-1_mtiv=0_lr=0.0001_w2v=g	REC	0.7734000087	0
RNN_n=2_h=150_d=0__mvs=-1_mtiv=0_lr=0.0001_w2v=g	REC	0.7734000087	0
LSTM_n=3_h=150_d=0.9__mvs=-1_mtiv=0_lr=0.0001_w2v=g	REC	0.7710000277	0
LSTM_n=3_h=50_d=0__mvs=-1_mtiv=0_lr=0.0001_w2v=g	REC	0.7710000277	0
LSTM_n=3_h=150_d=0.5__mvs=-1_mtiv=0_lr=0.0001_w2v=g	REC	0.7687000036	0
RNN_n=2_h=50_d=0__mvs=-1_mtiv=0_lr=0.0001_w2v=g	REC	0.7687000036	0
LSTM_n=5_h=50_d=0.9__mvs=-1_mtiv=0_lr=0.0001_w2v=g	REC	0.7663999796	0
GRU_n=3_h=300_d=0__mvs=-1_mtiv=0_lr=0.0001_w2v=g	REC	0.7663599968	0.04649267824
GRU_n=3_h=300_d=0.9__mvs=-1_mtiv=0_lr=0.0001_w2v=g	REC	0.7644999862	0.02275073339
RNN_n=2_h=150_d=0.9__mvs=-1_mtiv=0_lr=0.0001_w2v=g	REC	0.7639999986	0
GRU_n=2_h=300_d=0.5__mvs=-1_mtiv=0_lr=0.0001_w2v=g	REC	0.7635599852	0.02608559495
GRU_n=2_h=50_d=0__mvs=-1_mtiv=0_lr=0.0001_w2v=g	REC	0.7634250075	0.026331202
GRU_n=5_h=300_d=0__mvs=-1_mtiv=0_lr=0.0001_w2v=g	REC	0.7626199841	0.02419144255
GRU_n=2_h=300_d=0__mvs=-1_mtiv=0_lr=0.0001_w2v=g	REC	0.7621600032	0.03349999628
GRU_n=3_h=300_d=0.5__mvs=-1_mtiv=0_lr=0.0001_w2v=g	REC	0.7621399999	0.01561454964
RNN_n=5_h=50_d=0__mvs=-1_mtiv=0_lr=0.0001_w2v=g	REC	0.7570000291	0
GRU_n=5_h=300_d=0.5__mvs=-1_mtiv=0_lr=0.0001_w2v=g	REC	0.7547000051	0
BASELINE__mvs=-1_mtiv=0_lr=0.0001_w2v=g	REC	0.7523399949	0.01666523091
RNN_n=3_h=150_d=0__mvs=-1_mtiv=0_lr=0.0001_w2v=g	REC	0.752300024	0
RNN_n=3_h=50_d=0.5__mvs=-1_mtiv=0_lr=0.0001_w2v=g	REC	0.752300024	0
GRU_n=3_h=150_d=0.5__mvs=-1_mtiv=0_lr=0.0001_w2v=g	REC	0.75	0
RNN_n=5_h=300_d=0.5__mvs=-1_mtiv=0_lr=0.0001_w2v=g	REC	0.75	0
GRU_n=2_h=300_d=0.9__mvs=-1_mtiv=0_lr=0.0001_w2v=g	REC	0.7490799785	0.01657050111
GRU_n=5_h=50_d=0.9__mvs=-1_mtiv=0_lr=0.0001_w2v=g	REC	0.747699976	0
LSTM_n=5_h=150_d=0__mvs=-1_mtiv=0_lr=0.0001_w2v=g	REC	0.747699976	0
LSTM_n=3_h=50_d=0.5__mvs=-1_mtiv=0_lr=0.0001_w2v=g	REC	0.7452999949	0
RNN_n=5_h=150_d=0.5__mvs=-1_mtiv=0_lr=0.0001_w2v=g	REC	0.7383000255	0
LSTM_n=2_h=300_d=0.5__mvs=-1_mtiv=0_lr=0.0001_w2v=g	REC	0.7220000029	0
RNN_n=2_h=300_d=0.5__mvs=-1_mtiv=0_lr=0.0001_w2v=g	REC	0.7196000218	0
RNN_n=5_h=300_d=0__mvs=-1_mtiv=0_lr=0.0001_w2v=g	REC	0.7196000218	0
RNN_n=3_h=300_d=0.9__mvs=-1_mtiv=0_lr=0.0001_w2v=g	REC	0.7172999978	0
RNN_n=5_h=150_d=0__mvs=-1_mtiv=0_lr=0.0001_w2v=g	REC	0.7078999877	0
GRU_n=5_h=150_d=0__mvs=-1_mtiv=0_lr=0.0001_w2v=g	REC	0.7056000233	0
GRU_n=3_h=150_d=0__mvs=-1_mtiv=0_lr=0.0001_w2v=g	REC	0.6798999906	0
RNN_n=3_h=300_d=0__mvs=-1_mtiv=0_lr=0.0001_w2v=g	REC	0.5957999825	0
RNN_n=5_h=150_d=0.9__mvs=-1_mtiv=0_lr=0.0001_w2v=g	REC	0	0
RNN_n=5_h=300_d=0.9__mvs=-1_mtiv=0_lr=0.0001_w2v=g	REC	0	0
RNN_n=5_h=50_d=0.9__mvs=-1_mtiv=0_lr=0.0001_w2v=g	REC	0	0
```
</details>

Točnost veću od 0.814 postizu vecinom modeli od pet slojeva i s veličinom skrivenog sloja od 150 ili 300. U vrhu se našla i jedna konfiguracija s 2 sloja. Prva četiri modela s najvišom točnošću korsite dropout.

Isječak:
```js
LSTM_n=5_h=300_d=0.5__mvs=-1_mtiv=0_lr=0.0001_w2v=g	ACC	0.8234000206
GRU_n=5_h=150_d=0.5__mvs=-1_mtiv=0_lr=0.0001_w2v=g	ACC	0.8187999725
GRU_n=2_h=150_d=0.5__mvs=-1_mtiv=0_lr=0.0001_w2v=g	ACC	0.8165000081
LSTM_n=5_h=150_d=0.5__mvs=-1_mtiv=0_lr=0.0001_w2v=g	ACC	0.8141999841
LSTM_n=2_h=300_d=0__mvs=-1_mtiv=0_lr=0.0001_w2v=g	ACC	0.8130999804
GRU_n=5_h=300_d=0__mvs=-1_mtiv=0_lr=0.0001_w2v=g	ACC	0.8126199961
LSTM_n=2_h=50_d=0.9__mvs=-1_mtiv=0_lr=0.0001_w2v=g	ACC	0.8119000196
LSTM_n=3_h=150_d=0__mvs=-1_mtiv=0_lr=0.0001_w2v=g	ACC	0.8119000196
LSTM_n=3_h=150_d=0.5__mvs=-1_mtiv=0_lr=0.0001_w2v=g	ACC	0.8119000196
```

Najniži loss su postigli modeli s širim skrivenim slojem i s više slojeva:
```js
GRU_n=5_h=150_d=0__mvs=-1_mtiv=0_lr=0.0001_w2v=g	LOSS	0.4210999906
GRU_n=5_h=300_d=0__mvs=-1_mtiv=0_lr=0.0001_w2v=g	LOSS	0.4228799999
LSTM_n=5_h=300_d=0.5__mvs=-1_mtiv=0_lr=0.0001_w2v=g	LOSS	0.4257000089
GRU_n=2_h=300_d=0.9__mvs=-1_mtiv=0_lr=0.0001_w2v=g	LOSS	0.4263999999
GRU_n=5_h=300_d=0.5__mvs=-1_mtiv=0_lr=0.0001_w2v=g	LOSS	0.4271000028
LSTM_n=2_h=300_d=0.5__mvs=-1_mtiv=0_lr=0.0001_w2v=g	LOSS	0.4298000038
GRU_n=2_h=300_d=0__mvs=-1_mtiv=0_lr=0.0001_w2v=g	LOSS	0.4298200071
LSTM_n=2_h=300_d=0__mvs=-1_mtiv=0_lr=0.0001_w2v=g	LOSS	0.4300999939
LSTM_n=2_h=300_d=0.9__mvs=-1_mtiv=0_lr=0.0001_w2v=g	LOSS	0.4302000105
GRU_n=3_h=300_d=0__mvs=-1_mtiv=0_lr=0.0001_w2v=g	LOSS	0.4315999985
```

LSTMu je dropout od 50% jako pomogao i omogućio da postigne najbolji rezultat od svih (iako je moguće da je to bio samo statistički slučajan uspjeh jer nisam mjerio drugi moment):
```js
LSTM_n=5_h=300_d=0__mvs=-1_mtiv=0_lr=0.0001_w2v=g	ACC	0.787800014
LSTM_n=5_h=300_d=0.5__mvs=-1_mtiv=0_lr=0.0001_w2v=g	ACC	0.8234000206
LSTM_n=5_h=300_d=0.9__mvs=-1_mtiv=0_lr=0.0001_w2v=g	ACC	0.8004999757
```

Za dublje RNN ćelije je visok dropout značajno pogoršao rezutlate.

Isječak:
```js
RNN_n=5_h=150_d=0__mvs=-1_mtiv=0_lr=0.0001_w2v=g	ACC	0.7810000181
RNN_n=5_h=150_d=0.5__mvs=-1_mtiv=0_lr=0.0001_w2v=g	ACC	0.7900999784
RNN_n=5_h=150_d=0.9__mvs=-1_mtiv=0_lr=0.0001_w2v=g	ACC	0.5091999769

RNN_n=5_h=300_d=0__mvs=-1_mtiv=0_lr=0.0001_w2v=g	ACC	0.7786999941
RNN_n=5_h=300_d=0.5__mvs=-1_mtiv=0_lr=0.0001_w2v=g	ACC	0.7797999978
RNN_n=5_h=300_d=0.9__mvs=-1_mtiv=0_lr=0.0001_w2v=g	ACC	0.5091999769

RNN_n=5_h=50_d=0__mvs=-1_mtiv=0_lr=0.0001_w2v=g	ACC	0.787800014
RNN_n=5_h=50_d=0.5__mvs=-1_mtiv=0_lr=0.0001_w2v=g	ACC	0.799300015
RNN_n=5_h=50_d=0.9__mvs=-1_mtiv=0_lr=0.0001_w2v=g	ACC	0.5091999769
```

Nije jasno koji je model među testiranim najbolji, osobito kada ne postoje mjere drugog momenta. LSTM konfiguracija ***LSTM_n=5_h=300_d=0.5__mvs=-1_mtiv=0_lr=0.0001_w2v=g postiže*** najvišu točnost koja je za $0.5\%$ bolja od druge najbolje konfiguracije (***GRU_n=5_h=150_d=0.5__mvs=-1_mtiv=0_lr=0.0001_w2v=g***) i treći najniži gubitak pa ću izabrati njega za drugi eksperiment.

## Eksperiment 2

Proveo sam eksperimente za Baseline i najbolji LSTM model s sljedećim konfiguracijama:

- max_vocab_size_list $\in \{ -1, 100, 200, 1000 \}$
- min_token_freq_in_vocab_list $\in \{ 0, 20, 200, 2000 \}$
- word2vec_fn_list $\in \{ \mathrm{GLOVE}, \mathrm{RANDOM} \}$

Zbog nedostataka računalnih resursa nisam htio pokretati više kombinacija od ovoga. Ovo daje $4*4*2*2=64$ kombinacije. Svaku sam pokrenuo samo 3 puta za određuvanje procjene standardne devijacije.

Nisam siguran zašto na baseline `max_vocab_size_list` i `min_token_freq_in_vocab_list` nemaju nikakvog utjecaja. Moguće da je riječ o nekom kukcu.

<details>

<summary> Eksperiment 1 -- svi rezultati </summary>
```js
run_name	metric	value	stddev (out of 3)
LSTM_n=5_h=300_d=0.5__mvs=100_mtiv=20_lr=0.0001_w2v=g	ACC	0.8173000018	0.00196128629
LSTM_n=5_h=300_d=0.5__mvs=-1_mtiv=0_lr=0.0001_w2v=g	ACC	0.8165000081	0.004654024126
LSTM_n=5_h=300_d=0.5__mvs=-1_mtiv=2000_lr=0.0001_w2v=g	ACC	0.8164999882	0.006771033423
LSTM_n=5_h=300_d=0.5__mvs=200_mtiv=20_lr=0.0001_w2v=g	ACC	0.8145666718	0.007270659256
LSTM_n=5_h=300_d=0.5__mvs=1000_mtiv=2000_lr=0.0001_w2v=g	ACC	0.8142333428	0.004083584757
LSTM_n=5_h=300_d=0.5__mvs=1000_mtiv=200_lr=0.0001_w2v=g	ACC	0.813833336	0.007552629056
LSTM_n=5_h=300_d=0.5__mvs=100_mtiv=2000_lr=0.0001_w2v=g	ACC	0.813833336	0.01002741036
LSTM_n=5_h=300_d=0.5__mvs=1000_mtiv=20_lr=0.0001_w2v=g	ACC	0.8134333293	0.003909234591
LSTM_n=5_h=300_d=0.5__mvs=-1_mtiv=200_lr=0.0001_w2v=g	ACC	0.8119333386	0.00657283538
LSTM_n=5_h=300_d=0.5__mvs=-1_mtiv=20_lr=0.0001_w2v=g	ACC	0.8107999961	0.001877937425
LSTM_n=5_h=300_d=0.5__mvs=100_mtiv=200_lr=0.0001_w2v=g	ACC	0.8100333412	0.005737220151
LSTM_n=5_h=300_d=0.5__mvs=200_mtiv=0_lr=0.0001_w2v=g	ACC	0.8096333543	0.002468913681
LSTM_n=5_h=300_d=0.5__mvs=100_mtiv=0_lr=0.0001_w2v=g	ACC	0.8096333345	0.005593054687
LSTM_n=5_h=300_d=0.5__mvs=200_mtiv=200_lr=0.0001_w2v=g	ACC	0.8084999919	0.005633836609
LSTM_n=5_h=300_d=0.5__mvs=200_mtiv=2000_lr=0.0001_w2v=g	ACC	0.8080999851	0.003798237318
LSTM_n=5_h=300_d=0.5__mvs=1000_mtiv=0_lr=0.0001_w2v=g	ACC	0.8061666489	0.00706744915
Baseline__mvs=-1_mtiv=0_lr=0.0001_w2v=g	ACC	0.7824666699	0.005322487741
Baseline__mvs=200_mtiv=0_lr=0.0001_w2v=g	ACC	0.7824666699	0.005322487741
Baseline__mvs=1000_mtiv=0_lr=0.0001_w2v=g	ACC	0.7824666699	0.005322487741
Baseline__mvs=-1_mtiv=200_lr=0.0001_w2v=g	ACC	0.7824666699	0.005322487741
Baseline__mvs=200_mtiv=200_lr=0.0001_w2v=g	ACC	0.7824666699	0.005322487741
Baseline__mvs=1000_mtiv=200_lr=0.0001_w2v=g	ACC	0.7824666699	0.005322487741
Baseline__mvs=-1_mtiv=20_lr=0.0001_w2v=g	ACC	0.7824666699	0.005322487741
Baseline__mvs=200_mtiv=20_lr=0.0001_w2v=g	ACC	0.7824666699	0.005322487741
Baseline__mvs=1000_mtiv=20_lr=0.0001_w2v=g	ACC	0.7824666699	0.005322487741
Baseline__mvs=-1_mtiv=2000_lr=0.0001_w2v=g	ACC	0.7824666699	0.005322487741
Baseline__mvs=200_mtiv=2000_lr=0.0001_w2v=g	ACC	0.7824666699	0.005322487741
Baseline__mvs=1000_mtiv=2000_lr=0.0001_w2v=g	ACC	0.7824666699	0.005322487741
Baseline__mvs=100_mtiv=0_lr=0.0001_w2v=g	ACC	0.7824666699	0.005322487741
Baseline__mvs=100_mtiv=20_lr=0.0001_w2v=g	ACC	0.7824666699	0.005322487741
Baseline__mvs=100_mtiv=200_lr=0.0001_w2v=g	ACC	0.7824666699	0.005322487741
Baseline__mvs=100_mtiv=2000_lr=0.0001_w2v=g	ACC	0.7824666699	0.005322487741
LSTM_n=5_h=300_d=0.5__mvs=1000_mtiv=20_lr=0.0001_w2v=r	ACC	0.6708666484	0.04947593357
LSTM_n=5_h=300_d=0.5__mvs=200_mtiv=200_lr=0.0001_w2v=r	ACC	0.6647666693	0.03966649048
LSTM_n=5_h=300_d=0.5__mvs=-1_mtiv=0_lr=0.0001_w2v=r	ACC	0.6555666725	0.02980485122
LSTM_n=5_h=300_d=0.5__mvs=100_mtiv=20_lr=0.0001_w2v=r	ACC	0.6528666615	0.03077763663
LSTM_n=5_h=300_d=0.5__mvs=1000_mtiv=200_lr=0.0001_w2v=r	ACC	0.6429666479	0.05311881573
LSTM_n=5_h=300_d=0.5__mvs=1000_mtiv=0_lr=0.0001_w2v=r	ACC	0.6407000025	0.04893425514
LSTM_n=5_h=300_d=0.5__mvs=100_mtiv=200_lr=0.0001_w2v=r	ACC	0.6402666767	0.04039020147
LSTM_n=5_h=300_d=0.5__mvs=-1_mtiv=200_lr=0.0001_w2v=r	ACC	0.6402666569	0.0402818755
LSTM_n=5_h=300_d=0.5__mvs=200_mtiv=0_lr=0.0001_w2v=r	ACC	0.6398666501	0.0405408007
LSTM_n=5_h=300_d=0.5__mvs=100_mtiv=0_lr=0.0001_w2v=r	ACC	0.6391333143	0.04028403049
LSTM_n=5_h=300_d=0.5__mvs=-1_mtiv=2000_lr=0.0001_w2v=r	ACC	0.6376000047	0.0430145053
LSTM_n=5_h=300_d=0.5__mvs=200_mtiv=20_lr=0.0001_w2v=r	ACC	0.6345666647	0.04525796893
LSTM_n=5_h=300_d=0.5__mvs=-1_mtiv=20_lr=0.0001_w2v=r	ACC	0.631099999	0.03793494234
LSTM_n=5_h=300_d=0.5__mvs=1000_mtiv=2000_lr=0.0001_w2v=r	ACC	0.6307333509	0.04303148032
LSTM_n=5_h=300_d=0.5__mvs=200_mtiv=2000_lr=0.0001_w2v=r	ACC	0.6295666695	0.03166169292
LSTM_n=5_h=300_d=0.5__mvs=100_mtiv=2000_lr=0.0001_w2v=r	ACC	0.6246000131	0.03237726533
Baseline__mvs=-1_mtiv=0_lr=0.0001_w2v=r	ACC	0.5783666571	0.05442783907
Baseline__mvs=200_mtiv=0_lr=0.0001_w2v=r	ACC	0.5783666571	0.05442783907
Baseline__mvs=1000_mtiv=0_lr=0.0001_w2v=r	ACC	0.5783666571	0.05442783907
Baseline__mvs=-1_mtiv=200_lr=0.0001_w2v=r	ACC	0.5783666571	0.05442783907
Baseline__mvs=200_mtiv=200_lr=0.0001_w2v=r	ACC	0.5783666571	0.05442783907
Baseline__mvs=1000_mtiv=200_lr=0.0001_w2v=r	ACC	0.5783666571	0.05442783907
Baseline__mvs=-1_mtiv=20_lr=0.0001_w2v=r	ACC	0.5783666571	0.05442783907
Baseline__mvs=200_mtiv=20_lr=0.0001_w2v=r	ACC	0.5783666571	0.05442783907
Baseline__mvs=1000_mtiv=20_lr=0.0001_w2v=r	ACC	0.5783666571	0.05442783907
Baseline__mvs=-1_mtiv=2000_lr=0.0001_w2v=r	ACC	0.5783666571	0.05442783907
Baseline__mvs=200_mtiv=2000_lr=0.0001_w2v=r	ACC	0.5783666571	0.05442783907
Baseline__mvs=1000_mtiv=2000_lr=0.0001_w2v=r	ACC	0.5783666571	0.05442783907
Baseline__mvs=100_mtiv=0_lr=0.0001_w2v=r	ACC	0.5783666571	0.05442783907
Baseline__mvs=100_mtiv=20_lr=0.0001_w2v=r	ACC	0.5783666571	0.05442783907
Baseline__mvs=100_mtiv=200_lr=0.0001_w2v=r	ACC	0.5783666571	0.05442783907
Baseline__mvs=100_mtiv=2000_lr=0.0001_w2v=r	ACC	0.5783666571	0.05442783907
Baseline__mvs=-1_mtiv=0_lr=0.0001_w2v=r	F1	nan	nan
Baseline__mvs=200_mtiv=0_lr=0.0001_w2v=r	F1	nan	nan
Baseline__mvs=1000_mtiv=0_lr=0.0001_w2v=r	F1	nan	nan
Baseline__mvs=-1_mtiv=200_lr=0.0001_w2v=r	F1	nan	nan
Baseline__mvs=200_mtiv=200_lr=0.0001_w2v=r	F1	nan	nan
Baseline__mvs=1000_mtiv=200_lr=0.0001_w2v=r	F1	nan	nan
Baseline__mvs=-1_mtiv=20_lr=0.0001_w2v=r	F1	nan	nan
Baseline__mvs=200_mtiv=20_lr=0.0001_w2v=r	F1	nan	nan
Baseline__mvs=1000_mtiv=20_lr=0.0001_w2v=r	F1	nan	nan
Baseline__mvs=-1_mtiv=2000_lr=0.0001_w2v=r	F1	nan	nan
Baseline__mvs=200_mtiv=2000_lr=0.0001_w2v=r	F1	nan	nan
Baseline__mvs=1000_mtiv=2000_lr=0.0001_w2v=r	F1	nan	nan
Baseline__mvs=100_mtiv=0_lr=0.0001_w2v=r	F1	nan	nan
Baseline__mvs=100_mtiv=20_lr=0.0001_w2v=r	F1	nan	nan
Baseline__mvs=100_mtiv=200_lr=0.0001_w2v=r	F1	nan	nan
Baseline__mvs=100_mtiv=2000_lr=0.0001_w2v=r	F1	nan	nan
LSTM_n=5_h=300_d=0.5__mvs=-1_mtiv=2000_lr=0.0001_w2v=g	F1	0.8112333218	0.00184452411
LSTM_n=5_h=300_d=0.5__mvs=100_mtiv=20_lr=0.0001_w2v=g	F1	0.8108999928	0.002011624913
LSTM_n=5_h=300_d=0.5__mvs=1000_mtiv=20_lr=0.0001_w2v=g	F1	0.8056666652	0.005342490591
LSTM_n=5_h=300_d=0.5__mvs=200_mtiv=20_lr=0.0001_w2v=g	F1	0.8039000034	0.009697770937
LSTM_n=5_h=300_d=0.5__mvs=1000_mtiv=2000_lr=0.0001_w2v=g	F1	0.8030666709	0.0006018476366
LSTM_n=5_h=300_d=0.5__mvs=-1_mtiv=200_lr=0.0001_w2v=g	F1	0.8028333187	0.01251568637
LSTM_n=5_h=300_d=0.5__mvs=-1_mtiv=0_lr=0.0001_w2v=g	F1	0.8027999997	0.008580199771
LSTM_n=5_h=300_d=0.5__mvs=1000_mtiv=200_lr=0.0001_w2v=g	F1	0.8026333253	0.01201592388
LSTM_n=5_h=300_d=0.5__mvs=100_mtiv=200_lr=0.0001_w2v=g	F1	0.8025666475	0.002644913963
LSTM_n=5_h=300_d=0.5__mvs=1000_mtiv=0_lr=0.0001_w2v=g	F1	0.7998999953	0.002237543237
LSTM_n=5_h=300_d=0.5__mvs=-1_mtiv=20_lr=0.0001_w2v=g	F1	0.7990333239	0.005972334597
LSTM_n=5_h=300_d=0.5__mvs=200_mtiv=2000_lr=0.0001_w2v=g	F1	0.7990333239	0.007259628076
LSTM_n=5_h=300_d=0.5__mvs=100_mtiv=0_lr=0.0001_w2v=g	F1	0.7990000049	0.009156782956
LSTM_n=5_h=300_d=0.5__mvs=100_mtiv=2000_lr=0.0001_w2v=g	F1	0.7982999881	0.01211032602
LSTM_n=5_h=300_d=0.5__mvs=200_mtiv=0_lr=0.0001_w2v=g	F1	0.795966665	0.006298322445
LSTM_n=5_h=300_d=0.5__mvs=200_mtiv=200_lr=0.0001_w2v=g	F1	0.795599997	0.009200380385
Baseline__mvs=-1_mtiv=0_lr=0.0001_w2v=g	F1	0.7721999884	0.007654187341
Baseline__mvs=200_mtiv=0_lr=0.0001_w2v=g	F1	0.7721999884	0.007654187341
Baseline__mvs=1000_mtiv=0_lr=0.0001_w2v=g	F1	0.7721999884	0.007654187341
Baseline__mvs=-1_mtiv=200_lr=0.0001_w2v=g	F1	0.7721999884	0.007654187341
Baseline__mvs=200_mtiv=200_lr=0.0001_w2v=g	F1	0.7721999884	0.007654187341
Baseline__mvs=1000_mtiv=200_lr=0.0001_w2v=g	F1	0.7721999884	0.007654187341
Baseline__mvs=-1_mtiv=20_lr=0.0001_w2v=g	F1	0.7721999884	0.007654187341
Baseline__mvs=200_mtiv=20_lr=0.0001_w2v=g	F1	0.7721999884	0.007654187341
Baseline__mvs=1000_mtiv=20_lr=0.0001_w2v=g	F1	0.7721999884	0.007654187341
Baseline__mvs=-1_mtiv=2000_lr=0.0001_w2v=g	F1	0.7721999884	0.007654187341
Baseline__mvs=200_mtiv=2000_lr=0.0001_w2v=g	F1	0.7721999884	0.007654187341
Baseline__mvs=1000_mtiv=2000_lr=0.0001_w2v=g	F1	0.7721999884	0.007654187341
Baseline__mvs=100_mtiv=0_lr=0.0001_w2v=g	F1	0.7721999884	0.007654187341
Baseline__mvs=100_mtiv=20_lr=0.0001_w2v=g	F1	0.7721999884	0.007654187341
Baseline__mvs=100_mtiv=200_lr=0.0001_w2v=g	F1	0.7721999884	0.007654187341
Baseline__mvs=100_mtiv=2000_lr=0.0001_w2v=g	F1	0.7721999884	0.007654187341
LSTM_n=5_h=300_d=0.5__mvs=1000_mtiv=20_lr=0.0001_w2v=r	F1	0.6005666852	0.07962957963
LSTM_n=5_h=300_d=0.5__mvs=200_mtiv=200_lr=0.0001_w2v=r	F1	0.5870666603	0.06601213225
LSTM_n=5_h=300_d=0.5__mvs=100_mtiv=20_lr=0.0001_w2v=r	F1	0.5708333254	0.06403459371
LSTM_n=5_h=300_d=0.5__mvs=-1_mtiv=0_lr=0.0001_w2v=r	F1	0.5673999886	0.05886939052
LSTM_n=5_h=300_d=0.5__mvs=1000_mtiv=0_lr=0.0001_w2v=r	F1	0.5291333199	0.1074342081
LSTM_n=5_h=300_d=0.5__mvs=1000_mtiv=200_lr=0.0001_w2v=r	F1	0.52396667	0.1155292787
LSTM_n=5_h=300_d=0.5__mvs=100_mtiv=0_lr=0.0001_w2v=r	F1	0.5210666656	0.09444987091
LSTM_n=5_h=300_d=0.5__mvs=-1_mtiv=200_lr=0.0001_w2v=r	F1	0.5172666609	0.08869634033
LSTM_n=5_h=300_d=0.5__mvs=100_mtiv=200_lr=0.0001_w2v=r	F1	0.5171999931	0.09505263599
LSTM_n=5_h=300_d=0.5__mvs=-1_mtiv=20_lr=0.0001_w2v=r	F1	0.5160666605	0.09118425856
LSTM_n=5_h=300_d=0.5__mvs=-1_mtiv=2000_lr=0.0001_w2v=r	F1	0.5116333365	0.101488403
LSTM_n=5_h=300_d=0.5__mvs=200_mtiv=0_lr=0.0001_w2v=r	F1	0.5101666749	0.1018323652
LSTM_n=5_h=300_d=0.5__mvs=100_mtiv=2000_lr=0.0001_w2v=r	F1	0.5089666645	0.1132879043
LSTM_n=5_h=300_d=0.5__mvs=200_mtiv=2000_lr=0.0001_w2v=r	F1	0.506400009	0.081870925
LSTM_n=5_h=300_d=0.5__mvs=200_mtiv=20_lr=0.0001_w2v=r	F1	0.5049666663	0.1072340164
LSTM_n=5_h=300_d=0.5__mvs=1000_mtiv=2000_lr=0.0001_w2v=r	F1	0.4929333429	0.1042370535
Baseline__mvs=-1_mtiv=0_lr=0.0001_w2v=r	LOSS	0.6461333235	0.02015677146
Baseline__mvs=200_mtiv=0_lr=0.0001_w2v=r	LOSS	0.6461333235	0.02015677146
Baseline__mvs=1000_mtiv=0_lr=0.0001_w2v=r	LOSS	0.6461333235	0.02015677146
Baseline__mvs=-1_mtiv=200_lr=0.0001_w2v=r	LOSS	0.6461333235	0.02015677146
Baseline__mvs=200_mtiv=200_lr=0.0001_w2v=r	LOSS	0.6461333235	0.02015677146
Baseline__mvs=1000_mtiv=200_lr=0.0001_w2v=r	LOSS	0.6461333235	0.02015677146
Baseline__mvs=-1_mtiv=20_lr=0.0001_w2v=r	LOSS	0.6461333235	0.02015677146
Baseline__mvs=200_mtiv=20_lr=0.0001_w2v=r	LOSS	0.6461333235	0.02015677146
Baseline__mvs=1000_mtiv=20_lr=0.0001_w2v=r	LOSS	0.6461333235	0.02015677146
Baseline__mvs=-1_mtiv=2000_lr=0.0001_w2v=r	LOSS	0.6461333235	0.02015677146
Baseline__mvs=200_mtiv=2000_lr=0.0001_w2v=r	LOSS	0.6461333235	0.02015677146
Baseline__mvs=1000_mtiv=2000_lr=0.0001_w2v=r	LOSS	0.6461333235	0.02015677146
Baseline__mvs=100_mtiv=0_lr=0.0001_w2v=r	LOSS	0.6461333235	0.02015677146
Baseline__mvs=100_mtiv=20_lr=0.0001_w2v=r	LOSS	0.6461333235	0.02015677146
Baseline__mvs=100_mtiv=200_lr=0.0001_w2v=r	LOSS	0.6461333235	0.02015677146
Baseline__mvs=100_mtiv=2000_lr=0.0001_w2v=r	LOSS	0.6461333235	0.02015677146
LSTM_n=5_h=300_d=0.5__mvs=100_mtiv=2000_lr=0.0001_w2v=r	LOSS	0.630400002	0.01270510579
LSTM_n=5_h=300_d=0.5__mvs=200_mtiv=0_lr=0.0001_w2v=r	LOSS	0.6291666826	0.01493593694
LSTM_n=5_h=300_d=0.5__mvs=200_mtiv=20_lr=0.0001_w2v=r	LOSS	0.6287666758	0.01309561462
LSTM_n=5_h=300_d=0.5__mvs=1000_mtiv=0_lr=0.0001_w2v=r	LOSS	0.6284666657	0.01714842638
LSTM_n=5_h=300_d=0.5__mvs=100_mtiv=200_lr=0.0001_w2v=r	LOSS	0.6274000009	0.01620022471
LSTM_n=5_h=300_d=0.5__mvs=200_mtiv=2000_lr=0.0001_w2v=r	LOSS	0.6262999972	0.02045106876
LSTM_n=5_h=300_d=0.5__mvs=100_mtiv=0_lr=0.0001_w2v=r	LOSS	0.6254333258	0.0190175899
LSTM_n=5_h=300_d=0.5__mvs=1000_mtiv=2000_lr=0.0001_w2v=r	LOSS	0.6250666579	0.02303408291
LSTM_n=5_h=300_d=0.5__mvs=1000_mtiv=200_lr=0.0001_w2v=r	LOSS	0.6230666637	0.02561268236
LSTM_n=5_h=300_d=0.5__mvs=-1_mtiv=20_lr=0.0001_w2v=r	LOSS	0.6217333277	0.02371516815
LSTM_n=5_h=300_d=0.5__mvs=-1_mtiv=200_lr=0.0001_w2v=r	LOSS	0.6215999921	0.01933512264
LSTM_n=5_h=300_d=0.5__mvs=-1_mtiv=2000_lr=0.0001_w2v=r	LOSS	0.6213999987	0.02204783992
LSTM_n=5_h=300_d=0.5__mvs=-1_mtiv=0_lr=0.0001_w2v=r	LOSS	0.6130333344	0.02854589456
LSTM_n=5_h=300_d=0.5__mvs=200_mtiv=200_lr=0.0001_w2v=r	LOSS	0.6127666831	0.02038861077
LSTM_n=5_h=300_d=0.5__mvs=100_mtiv=20_lr=0.0001_w2v=r	LOSS	0.6127333244	0.0207271591
LSTM_n=5_h=300_d=0.5__mvs=1000_mtiv=20_lr=0.0001_w2v=r	LOSS	0.6043000023	0.02724271567
Baseline__mvs=-1_mtiv=0_lr=0.0001_w2v=g	LOSS	0.4706666668	0.002536831976
Baseline__mvs=200_mtiv=0_lr=0.0001_w2v=g	LOSS	0.4706666668	0.002536831976
Baseline__mvs=1000_mtiv=0_lr=0.0001_w2v=g	LOSS	0.4706666668	0.002536831976
Baseline__mvs=-1_mtiv=200_lr=0.0001_w2v=g	LOSS	0.4706666668	0.002536831976
Baseline__mvs=200_mtiv=200_lr=0.0001_w2v=g	LOSS	0.4706666668	0.002536831976
Baseline__mvs=1000_mtiv=200_lr=0.0001_w2v=g	LOSS	0.4706666668	0.002536831976
Baseline__mvs=-1_mtiv=20_lr=0.0001_w2v=g	LOSS	0.4706666668	0.002536831976
Baseline__mvs=200_mtiv=20_lr=0.0001_w2v=g	LOSS	0.4706666668	0.002536831976
Baseline__mvs=1000_mtiv=20_lr=0.0001_w2v=g	LOSS	0.4706666668	0.002536831976
Baseline__mvs=-1_mtiv=2000_lr=0.0001_w2v=g	LOSS	0.4706666668	0.002536831976
Baseline__mvs=200_mtiv=2000_lr=0.0001_w2v=g	LOSS	0.4706666668	0.002536831976
Baseline__mvs=1000_mtiv=2000_lr=0.0001_w2v=g	LOSS	0.4706666668	0.002536831976
Baseline__mvs=100_mtiv=0_lr=0.0001_w2v=g	LOSS	0.4706666668	0.002536831976
Baseline__mvs=100_mtiv=20_lr=0.0001_w2v=g	LOSS	0.4706666668	0.002536831976
Baseline__mvs=100_mtiv=200_lr=0.0001_w2v=g	LOSS	0.4706666668	0.002536831976
Baseline__mvs=100_mtiv=2000_lr=0.0001_w2v=g	LOSS	0.4706666668	0.002536831976
LSTM_n=5_h=300_d=0.5__mvs=-1_mtiv=2000_lr=0.0001_w2v=g	LOSS	0.4481333395	0.003235566154
LSTM_n=5_h=300_d=0.5__mvs=-1_mtiv=200_lr=0.0001_w2v=g	LOSS	0.4405000011	0.0106445646
LSTM_n=5_h=300_d=0.5__mvs=200_mtiv=20_lr=0.0001_w2v=g	LOSS	0.4393999974	0.005045781433
LSTM_n=5_h=300_d=0.5__mvs=100_mtiv=0_lr=0.0001_w2v=g	LOSS	0.4367666642	0.01020010467
LSTM_n=5_h=300_d=0.5__mvs=100_mtiv=20_lr=0.0001_w2v=g	LOSS	0.4365666608	0.01249035356
LSTM_n=5_h=300_d=0.5__mvs=-1_mtiv=0_lr=0.0001_w2v=g	LOSS	0.4340666731	0.01087944149
LSTM_n=5_h=300_d=0.5__mvs=1000_mtiv=0_lr=0.0001_w2v=g	LOSS	0.4338333309	0.00551503164
LSTM_n=5_h=300_d=0.5__mvs=200_mtiv=0_lr=0.0001_w2v=g	LOSS	0.4314666688	0.004836213961
LSTM_n=5_h=300_d=0.5__mvs=200_mtiv=2000_lr=0.0001_w2v=g	LOSS	0.4312333365	0.005756355358
LSTM_n=5_h=300_d=0.5__mvs=1000_mtiv=20_lr=0.0001_w2v=g	LOSS	0.4310333331	0.001211979201
LSTM_n=5_h=300_d=0.5__mvs=1000_mtiv=200_lr=0.0001_w2v=g	LOSS	0.4303333263	0.002590153436
LSTM_n=5_h=300_d=0.5__mvs=100_mtiv=200_lr=0.0001_w2v=g	LOSS	0.4300666551	0.005198933216
LSTM_n=5_h=300_d=0.5__mvs=1000_mtiv=2000_lr=0.0001_w2v=g	LOSS	0.428733329	0.004910080634
LSTM_n=5_h=300_d=0.5__mvs=100_mtiv=2000_lr=0.0001_w2v=g	LOSS	0.4283666611	0.004778653156
LSTM_n=5_h=300_d=0.5__mvs=-1_mtiv=20_lr=0.0001_w2v=g	LOSS	0.42566667	0.004374423413
LSTM_n=5_h=300_d=0.5__mvs=200_mtiv=200_lr=0.0001_w2v=g	LOSS	0.4249666731	0.004865071427
Baseline__mvs=-1_mtiv=0_lr=0.0001_w2v=r	PRE	nan	nan
Baseline__mvs=200_mtiv=0_lr=0.0001_w2v=r	PRE	nan	nan
Baseline__mvs=1000_mtiv=0_lr=0.0001_w2v=r	PRE	nan	nan
Baseline__mvs=-1_mtiv=200_lr=0.0001_w2v=r	PRE	nan	nan
Baseline__mvs=200_mtiv=200_lr=0.0001_w2v=r	PRE	nan	nan
Baseline__mvs=1000_mtiv=200_lr=0.0001_w2v=r	PRE	nan	nan
Baseline__mvs=-1_mtiv=20_lr=0.0001_w2v=r	PRE	nan	nan
Baseline__mvs=200_mtiv=20_lr=0.0001_w2v=r	PRE	nan	nan
Baseline__mvs=1000_mtiv=20_lr=0.0001_w2v=r	PRE	nan	nan
Baseline__mvs=-1_mtiv=2000_lr=0.0001_w2v=r	PRE	nan	nan
Baseline__mvs=200_mtiv=2000_lr=0.0001_w2v=r	PRE	nan	nan
Baseline__mvs=1000_mtiv=2000_lr=0.0001_w2v=r	PRE	nan	nan
Baseline__mvs=100_mtiv=0_lr=0.0001_w2v=r	PRE	nan	nan
Baseline__mvs=100_mtiv=20_lr=0.0001_w2v=r	PRE	nan	nan
Baseline__mvs=100_mtiv=200_lr=0.0001_w2v=r	PRE	nan	nan
Baseline__mvs=100_mtiv=2000_lr=0.0001_w2v=r	PRE	nan	nan
LSTM_n=5_h=300_d=0.5__mvs=100_mtiv=2000_lr=0.0001_w2v=g	PRE	0.8522666693	0.00745220421
LSTM_n=5_h=300_d=0.5__mvs=-1_mtiv=0_lr=0.0001_w2v=g	PRE	0.8492999872	0.009228574089
LSTM_n=5_h=300_d=0.5__mvs=200_mtiv=0_lr=0.0001_w2v=g	PRE	0.8397666613	0.0108250681
LSTM_n=5_h=300_d=0.5__mvs=1000_mtiv=2000_lr=0.0001_w2v=g	PRE	0.8379000028	0.01923902027
LSTM_n=5_h=300_d=0.5__mvs=1000_mtiv=200_lr=0.0001_w2v=g	PRE	0.8362999956	0.0126683161
LSTM_n=5_h=300_d=0.5__mvs=200_mtiv=20_lr=0.0001_w2v=g	PRE	0.8357000152	0.01056692538
LSTM_n=5_h=300_d=0.5__mvs=200_mtiv=200_lr=0.0001_w2v=g	PRE	0.8352333307	0.009711273516
LSTM_n=5_h=300_d=0.5__mvs=-1_mtiv=20_lr=0.0001_w2v=g	PRE	0.8345333338	0.01178313168
LSTM_n=5_h=300_d=0.5__mvs=100_mtiv=0_lr=0.0001_w2v=g	PRE	0.829700013	0.02000067323
LSTM_n=5_h=300_d=0.5__mvs=-1_mtiv=200_lr=0.0001_w2v=g	PRE	0.826699992	0.01686793637
LSTM_n=5_h=300_d=0.5__mvs=100_mtiv=20_lr=0.0001_w2v=g	PRE	0.8242333333	0.01076174524
LSTM_n=5_h=300_d=0.5__mvs=1000_mtiv=20_lr=0.0001_w2v=g	PRE	0.8241666754	0.003092290925
LSTM_n=5_h=300_d=0.5__mvs=200_mtiv=2000_lr=0.0001_w2v=g	PRE	0.8228666782	0.02150075906
LSTM_n=5_h=300_d=0.5__mvs=-1_mtiv=2000_lr=0.0001_w2v=g	PRE	0.8209000031	0.02416459013
LSTM_n=5_h=300_d=0.5__mvs=100_mtiv=200_lr=0.0001_w2v=g	PRE	0.8197999994	0.01801054653
LSTM_n=5_h=300_d=0.5__mvs=1000_mtiv=0_lr=0.0001_w2v=g	PRE	0.8122666677	0.02355651966
Baseline__mvs=-1_mtiv=0_lr=0.0001_w2v=g	PRE	0.7945333322	0.01044169453
Baseline__mvs=200_mtiv=0_lr=0.0001_w2v=g	PRE	0.7945333322	0.01044169453
Baseline__mvs=1000_mtiv=0_lr=0.0001_w2v=g	PRE	0.7945333322	0.01044169453
Baseline__mvs=-1_mtiv=200_lr=0.0001_w2v=g	PRE	0.7945333322	0.01044169453
Baseline__mvs=200_mtiv=200_lr=0.0001_w2v=g	PRE	0.7945333322	0.01044169453
Baseline__mvs=1000_mtiv=200_lr=0.0001_w2v=g	PRE	0.7945333322	0.01044169453
Baseline__mvs=-1_mtiv=20_lr=0.0001_w2v=g	PRE	0.7945333322	0.01044169453
Baseline__mvs=200_mtiv=20_lr=0.0001_w2v=g	PRE	0.7945333322	0.01044169453
Baseline__mvs=1000_mtiv=20_lr=0.0001_w2v=g	PRE	0.7945333322	0.01044169453
Baseline__mvs=-1_mtiv=2000_lr=0.0001_w2v=g	PRE	0.7945333322	0.01044169453
Baseline__mvs=200_mtiv=2000_lr=0.0001_w2v=g	PRE	0.7945333322	0.01044169453
Baseline__mvs=1000_mtiv=2000_lr=0.0001_w2v=g	PRE	0.7945333322	0.01044169453
Baseline__mvs=100_mtiv=0_lr=0.0001_w2v=g	PRE	0.7945333322	0.01044169453
Baseline__mvs=100_mtiv=20_lr=0.0001_w2v=g	PRE	0.7945333322	0.01044169453
Baseline__mvs=100_mtiv=200_lr=0.0001_w2v=g	PRE	0.7945333322	0.01044169453
Baseline__mvs=100_mtiv=2000_lr=0.0001_w2v=g	PRE	0.7945333322	0.01044169453
LSTM_n=5_h=300_d=0.5__mvs=200_mtiv=0_lr=0.0001_w2v=r	PRE	0.751300017	0.01588856073
LSTM_n=5_h=300_d=0.5__mvs=-1_mtiv=200_lr=0.0001_w2v=r	PRE	0.7468000054	0.03764678401
LSTM_n=5_h=300_d=0.5__mvs=100_mtiv=200_lr=0.0001_w2v=r	PRE	0.743599991	0.02226311646
LSTM_n=5_h=300_d=0.5__mvs=1000_mtiv=2000_lr=0.0001_w2v=r	PRE	0.7411666711	0.03666716891
LSTM_n=5_h=300_d=0.5__mvs=-1_mtiv=2000_lr=0.0001_w2v=r	PRE	0.7410666744	0.03326182295
LSTM_n=5_h=300_d=0.5__mvs=1000_mtiv=200_lr=0.0001_w2v=r	PRE	0.7386666735	0.04576201265
LSTM_n=5_h=300_d=0.5__mvs=100_mtiv=0_lr=0.0001_w2v=r	PRE	0.7367999951	0.038210291
LSTM_n=5_h=300_d=0.5__mvs=200_mtiv=20_lr=0.0001_w2v=r	PRE	0.73483332	0.02615623177
LSTM_n=5_h=300_d=0.5__mvs=-1_mtiv=0_lr=0.0001_w2v=r	PRE	0.7347666621	0.03356113631
LSTM_n=5_h=300_d=0.5__mvs=200_mtiv=200_lr=0.0001_w2v=r	PRE	0.7347000043	0.04044783075
LSTM_n=5_h=300_d=0.5__mvs=1000_mtiv=20_lr=0.0001_w2v=r	PRE	0.7320666711	0.05248533229
LSTM_n=5_h=300_d=0.5__mvs=1000_mtiv=0_lr=0.0001_w2v=r	PRE	0.7306666772	0.05420961586
LSTM_n=5_h=300_d=0.5__mvs=200_mtiv=2000_lr=0.0001_w2v=r	PRE	0.7233333389	0.0214625723
LSTM_n=5_h=300_d=0.5__mvs=100_mtiv=20_lr=0.0001_w2v=r	PRE	0.7227666577	0.04227847088
LSTM_n=5_h=300_d=0.5__mvs=-1_mtiv=20_lr=0.0001_w2v=r	PRE	0.7133333286	0.02522437843
LSTM_n=5_h=300_d=0.5__mvs=100_mtiv=2000_lr=0.0001_w2v=r	PRE	0.7055333257	0.04286297311
LSTM_n=5_h=300_d=0.5__mvs=-1_mtiv=2000_lr=0.0001_w2v=g	REC	0.8029666742	0.02059323945
LSTM_n=5_h=300_d=0.5__mvs=100_mtiv=20_lr=0.0001_w2v=g	REC	0.7982666691	0.01270834766
LSTM_n=5_h=300_d=0.5__mvs=1000_mtiv=0_lr=0.0001_w2v=g	REC	0.7889333169	0.01809353028
LSTM_n=5_h=300_d=0.5__mvs=1000_mtiv=20_lr=0.0001_w2v=g	REC	0.7881333232	0.01117326831
LSTM_n=5_h=300_d=0.5__mvs=100_mtiv=200_lr=0.0001_w2v=g	REC	0.7865999937	0.01267357798
LSTM_n=5_h=300_d=0.5__mvs=-1_mtiv=200_lr=0.0001_w2v=g	REC	0.7818999887	0.03589354181
LSTM_n=5_h=300_d=0.5__mvs=200_mtiv=2000_lr=0.0001_w2v=g	REC	0.778033336	0.02978392658
LSTM_n=5_h=300_d=0.5__mvs=200_mtiv=20_lr=0.0001_w2v=g	REC	0.7749333382	0.0210119565
LSTM_n=5_h=300_d=0.5__mvs=1000_mtiv=200_lr=0.0001_w2v=g	REC	0.7725999951	0.03039880146
LSTM_n=5_h=300_d=0.5__mvs=1000_mtiv=2000_lr=0.0001_w2v=g	REC	0.7718333205	0.01665658694
LSTM_n=5_h=300_d=0.5__mvs=100_mtiv=0_lr=0.0001_w2v=g	REC	0.7718000015	0.02972618922
LSTM_n=5_h=300_d=0.5__mvs=-1_mtiv=20_lr=0.0001_w2v=g	REC	0.7671333154	0.02110060671
LSTM_n=5_h=300_d=0.5__mvs=-1_mtiv=0_lr=0.0001_w2v=g	REC	0.7616666754	0.02199732935
LSTM_n=5_h=300_d=0.5__mvs=200_mtiv=200_lr=0.0001_w2v=g	REC	0.7601333261	0.02209455828
LSTM_n=5_h=300_d=0.5__mvs=200_mtiv=0_lr=0.0001_w2v=g	REC	0.7570000092	0.02011085225
Baseline__mvs=-1_mtiv=0_lr=0.0001_w2v=g	REC	0.7515333295	0.01910415557
Baseline__mvs=200_mtiv=0_lr=0.0001_w2v=g	REC	0.7515333295	0.01910415557
Baseline__mvs=1000_mtiv=0_lr=0.0001_w2v=g	REC	0.7515333295	0.01910415557
Baseline__mvs=-1_mtiv=200_lr=0.0001_w2v=g	REC	0.7515333295	0.01910415557
Baseline__mvs=200_mtiv=200_lr=0.0001_w2v=g	REC	0.7515333295	0.01910415557
Baseline__mvs=1000_mtiv=200_lr=0.0001_w2v=g	REC	0.7515333295	0.01910415557
Baseline__mvs=-1_mtiv=20_lr=0.0001_w2v=g	REC	0.7515333295	0.01910415557
Baseline__mvs=200_mtiv=20_lr=0.0001_w2v=g	REC	0.7515333295	0.01910415557
Baseline__mvs=1000_mtiv=20_lr=0.0001_w2v=g	REC	0.7515333295	0.01910415557
Baseline__mvs=-1_mtiv=2000_lr=0.0001_w2v=g	REC	0.7515333295	0.01910415557
Baseline__mvs=200_mtiv=2000_lr=0.0001_w2v=g	REC	0.7515333295	0.01910415557
Baseline__mvs=1000_mtiv=2000_lr=0.0001_w2v=g	REC	0.7515333295	0.01910415557
Baseline__mvs=100_mtiv=0_lr=0.0001_w2v=g	REC	0.7515333295	0.01910415557
Baseline__mvs=100_mtiv=20_lr=0.0001_w2v=g	REC	0.7515333295	0.01910415557
Baseline__mvs=100_mtiv=200_lr=0.0001_w2v=g	REC	0.7515333295	0.01910415557
Baseline__mvs=100_mtiv=2000_lr=0.0001_w2v=g	REC	0.7515333295	0.01910415557
LSTM_n=5_h=300_d=0.5__mvs=100_mtiv=2000_lr=0.0001_w2v=g	REC	0.7507666747	0.01670176053
LSTM_n=5_h=300_d=0.5__mvs=1000_mtiv=20_lr=0.0001_w2v=r	REC	0.5116999944	0.08841519663
LSTM_n=5_h=300_d=0.5__mvs=200_mtiv=200_lr=0.0001_w2v=r	REC	0.4906666577	0.07246338405
LSTM_n=5_h=300_d=0.5__mvs=100_mtiv=20_lr=0.0001_w2v=r	REC	0.4781666597	0.0848056588
LSTM_n=5_h=300_d=0.5__mvs=-1_mtiv=0_lr=0.0001_w2v=r	REC	0.4657333195	0.07143921807
LSTM_n=5_h=300_d=0.5__mvs=1000_mtiv=0_lr=0.0001_w2v=r	REC	0.4306999942	0.1308752757
LSTM_n=5_h=300_d=0.5__mvs=100_mtiv=2000_lr=0.0001_w2v=r	REC	0.426000009	0.1609281299
LSTM_n=5_h=300_d=0.5__mvs=1000_mtiv=200_lr=0.0001_w2v=r	REC	0.4228666623	0.1453258616
LSTM_n=5_h=300_d=0.5__mvs=-1_mtiv=20_lr=0.0001_w2v=r	REC	0.4151333272	0.1138762212
LSTM_n=5_h=300_d=0.5__mvs=100_mtiv=0_lr=0.0001_w2v=r	REC	0.4151000082	0.11637056
LSTM_n=5_h=300_d=0.5__mvs=100_mtiv=200_lr=0.0001_w2v=r	REC	0.4073333343	0.1156121193
LSTM_n=5_h=300_d=0.5__mvs=-1_mtiv=200_lr=0.0001_w2v=r	REC	0.4041999976	0.1024275503
LSTM_n=5_h=300_d=0.5__mvs=-1_mtiv=2000_lr=0.0001_w2v=r	REC	0.4034000039	0.1227731559
LSTM_n=5_h=300_d=0.5__mvs=200_mtiv=0_lr=0.0001_w2v=r	REC	0.400333335	0.1289150848
LSTM_n=5_h=300_d=0.5__mvs=200_mtiv=2000_lr=0.0001_w2v=r	REC	0.3987333377	0.101622255
LSTM_n=5_h=300_d=0.5__mvs=200_mtiv=20_lr=0.0001_w2v=r	REC	0.3987333278	0.1332794489
LSTM_n=5_h=300_d=0.5__mvs=1000_mtiv=2000_lr=0.0001_w2v=r	REC	0.3816333413	0.1197025604
Baseline__mvs=-1_mtiv=0_lr=0.0001_w2v=r	REC	0.2204333345	0.1822417137
Baseline__mvs=200_mtiv=0_lr=0.0001_w2v=r	REC	0.2204333345	0.1822417137
Baseline__mvs=1000_mtiv=0_lr=0.0001_w2v=r	REC	0.2204333345	0.1822417137
Baseline__mvs=-1_mtiv=200_lr=0.0001_w2v=r	REC	0.2204333345	0.1822417137
Baseline__mvs=200_mtiv=200_lr=0.0001_w2v=r	REC	0.2204333345	0.1822417137
Baseline__mvs=1000_mtiv=200_lr=0.0001_w2v=r	REC	0.2204333345	0.1822417137
Baseline__mvs=-1_mtiv=20_lr=0.0001_w2v=r	REC	0.2204333345	0.1822417137
Baseline__mvs=200_mtiv=20_lr=0.0001_w2v=r	REC	0.2204333345	0.1822417137
Baseline__mvs=1000_mtiv=20_lr=0.0001_w2v=r	REC	0.2204333345	0.1822417137
Baseline__mvs=-1_mtiv=2000_lr=0.0001_w2v=r	REC	0.2204333345	0.1822417137
Baseline__mvs=200_mtiv=2000_lr=0.0001_w2v=r	REC	0.2204333345	0.1822417137
Baseline__mvs=1000_mtiv=2000_lr=0.0001_w2v=r	REC	0.2204333345	0.1822417137
Baseline__mvs=100_mtiv=0_lr=0.0001_w2v=r	REC	0.2204333345	0.1822417137
Baseline__mvs=100_mtiv=20_lr=0.0001_w2v=r	REC	0.2204333345	0.1822417137
Baseline__mvs=100_mtiv=200_lr=0.0001_w2v=r	REC	0.2204333345	0.1822417137
Baseline__mvs=100_mtiv=2000_lr=0.0001_w2v=r	REC	0.2204333345	0.1822417137
```
</details>



> Probajte pokrenuti povratne neuronske mreže za najbolji set hiperparametara bez da koristite prednaučene vektorske reprezentacije. Probajte isto za vaš baseline model. Koji model više "pati" od gubitka prednaučenih reprezentacija?

Ne korišenje GLOVE embeddinga je značajno pogoršalo performansu i na Baselineu i na LSTM-u

