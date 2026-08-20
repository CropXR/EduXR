"""Generate the answer-model pages in docs/ from their student pages.

An answer model is the same interactive page with a worked answer inserted after
every block of open questions, following the pattern of the hand-built
mechanistic_model_answers.html. Keeping them generated means a student page and
its answer model cannot drift apart when the practical is edited, and the count
check below fails loudly if a question is ever added without an answer.

Run after any change to a student page:

    python tools/build_answers_page.py                 # rebuild every page
    python tools/build_answers_page.py go_enrichment   # rebuild just one
"""

import re
import sys
from pathlib import Path

DOCS = Path(__file__).resolve().parent.parent / "docs"

# One entry per <div class="callout q"> block in docs/deep_learning.html, in page order.
DEEP_LEARNING_ANSWERS = [

# --- Step 0: mechanistic vs data-driven -------------------------------------
"""<ul>
<li><b>Mechanistic vs. data-driven.</b> A question like <em>"how does a longer drought shift the trade-off between drought tolerance and pathogen defence?"</em> calls for a mechanistic model: you are asking <em>why</em>, you already have regulatory knowledge to encode, and you want to reason about interventions you have never tried. A question like <em>"which of these 40,000 field photographs show disease?"</em> calls for a data-driven model: nobody can write the rule down, but labelled examples are cheap. The deciding factors are whether you have mechanism you trust or data you trust, and whether you need an <em>explanation</em> or a <em>decision</em>.</li>
<li><b>Both at once.</b> It is a strength because no theory is required, so it works in areas where our understanding is incomplete, and it can pick up subtle cues a human could never put into words. It is a weakness because the model has no notion of what is <em>biologically</em> relevant: it cannot tell a causal cue (a lesion) from an incidental one (the lighting, the background, the camera). It cannot explain itself, cannot be checked against known biology, and will apply a nonsensical rule with complete confidence. Both statements are true simultaneously, which is what the rest of this practical is about.</li>
</ul>""",

# --- Step 1: the data and the numbers ---------------------------------------
"""<ul>
<li><b>What the numbers are.</b> Each individual number is the intensity of <em>one colour channel at one pixel position</em>. There are three per pixel because any colour can be reconstructed by mixing red, green and blue light, so the camera records how much of each is present. Cameras store these as whole numbers from 0 (none) to 255 (maximum); we divide by 255 so every input sits between 0 and 1, a range in which neural networks train much more stably. One leaf is therefore 64 &times; 64 &times; 3 = 12,288 numbers, plus a single label of 0 or 1.</li>
<li><b>What shrinking costs.</b> We throw away fine detail: small or early lesions, spore structures, fine vein texture, anything smaller than roughly a pixel at the new size. We do it because compute and memory scale with the number of pixels, and because fewer inputs means fewer things the model can latch onto, which matters a lot when you only have a few hundred training images. It is a trade-off: too small and the symptom itself disappears. At 64 &times; 64 the larger blotches and colour changes still survive.</li>
<li><b>The uniform background: both.</b> Helpful, because it removes a large source of irrelevant variation and lets the model concentrate on the leaf. Harmful, because it is unrealistically easy, a model trained on studio photographs against grey card has never seen soil, neighbouring leaves, shadows or direct sunlight, and will very likely fall over on a real field photo. It also means anything systematic about the <em>setup</em> rather than the leaf becomes available as a shortcut, which is precisely the trap in step 7.</li>
</ul>""",

# --- Step 2: train/test split -----------------------------------------------
"""<ul>
<li><b>Why split.</b> To measure generalisation. A model can score perfectly on its training images simply by memorising them, which tells you nothing. Only performance on data it never learned from estimates how it will behave on the next leaf you show it, and that is the only thing you actually care about.</li>
<li><b>What to consider.</b> Keep the class distribution representative in both halves, ideally stratifying so the proportions match. Leave the test set large enough for the estimate to be stable, and the training set large enough to learn from. Make sure near-duplicates cannot straddle the split, such as the same plant, the same leaf photographed twice, or the same field session. Split along the axis you actually want to generalise over: if the model must work on a new farm, hold out whole farms rather than random images. And do not tune your model repeatedly against the test set, or it quietly stops being held out.</li>
<li><b>The same plant in both halves.</b> Two photos of one plant are near-identical: same leaf shape, same lighting, same disease stage, same camera. If one is in training, the model can recognise its twin in the test set without having learned anything general, so your test score partly measures memorisation and comes out optimistically biased. This is called <b>data leakage</b>, and it is one of the most common reasons a model that looked excellent in development disappoints in the field.</li>
<li><b>Which number to quote.</b> The test-set number, always. The training number describes how well the model reproduces answers it was explicitly fitted to, so it is not evidence about anything unseen. Quoting it is not just optimistic, it is measuring the wrong thing: a model that had simply memorised every training image would score 100% there while being useless in a greenhouse.</li>
</ul>""",

# --- Step 3a: do you trust it? ----------------------------------------------
"""<ul>
<li><b>Do you trust it? You should not,</b> and the honest answer at this point is that you do not yet have enough information to say. That is the real lesson: a single accuracy figure is not evidence, whatever its value.</li>
<li><b>What to ask for.</b> A good list would include: how many of each class are there, and what does the trivial "always guess the most common class" baseline score? How does the model do on each class separately? What kinds of mistakes does it make, in a confusion matrix? How well does it rank cases, via AUROC? What does it actually predict on a handful of images I can look at? How was the test set constructed, and could it leak? If you asked for most of these, you were already thinking like a sceptic. The rest of this step works through them.</li>
</ul>""",

# --- Step 3b: the metrics ---------------------------------------------------
"""<ul>
<li><b>Does the distribution change your view? Completely.</b> About 90% of these leaves are infected, so the rule "always say infected" scores about 90% by construction. The meaningful comparison for any model is against that majority-class baseline, not against 50%. Move the slider to 50% healthy and the same model drops to about 0.50, which shows the score was a property of the dataset rather than of the model.</li>
<li><b>What the two plots show.</b> The <b>confusion matrix</b> is a table of true label against predicted label at <em>one chosen threshold</em>, splitting outcomes into true negatives, false positives, false negatives and true positives. It shows <em>which</em> mistakes are made, not just how many. Here the entire "predicted healthy" column is empty, because the model never predicts healthy. The <b>ROC curve</b> plots the true positive rate against the false positive rate across <em>every possible threshold</em>, so it describes the quality of the model's ranking independently of where you cut. Its area, the <b>AUROC</b>, is 1.0 for a perfect ranker and 0.5 for one carrying no information. Here the curve lies on the diagonal.</li>
<li><b>Useful predictions? None whatsoever.</b> It never identifies a single healthy leaf, its accuracy on the healthy class is 0.00, and its AUROC of 0.50 says it cannot separate the classes at all. It contains no information about its input: you would get identical output with the lens cap on.</li>
<li><b>The rule in one sentence:</b> "always answer infected", regardless of the image. (In the notebook this is <code>MagicModel</code>, whose <code>predict</code> literally returns an array of ones.)</li>
<li><b>Looking back at your list.</b> The thing most people miss first time is the class distribution, because accuracy feels like it should already account for it. Everything else here follows from that one question. If you would still deploy this model, ask yourself what it would do in a greenhouse where most plants are healthy: it would flag every single one.</li>
</ul>""",

# --- Step 4: training -------------------------------------------------------
"""<ul>
<li><b>What the outputs are.</b> A single number between 0 and 1 per image, produced by the final sigmoid. It expresses how strongly the network leans towards class 1 (infected): near 1 means confident infected, near 0 confident healthy. It is often loosely called a probability, but it only deserves that name if you have checked that it is calibrated, that among all images scored 0.7, roughly 70% really are infected.</li>
<li><b>A score near 0.5.</b> The model is undecided: whatever it extracted from the image does not clearly favour either class. In practice you would not want an automatic decision here at all. A sensible deployed system routes low-confidence cases to a human rather than forcing a call, which is often more valuable than squeezing out another percent of accuracy.</li>
<li><b>More epochs.</b> The training loss keeps falling and training accuracy keeps climbing, while the test curves flatten off and eventually turn the wrong way. The <em>gap</em> between the solid and dashed lines is overfitting made visible: it measures how much of the model's apparent skill is memorisation of these particular images rather than something that transfers. A widening gap is the signal to stop training, get more data, or regularise.</li>
</ul>""",

# --- Step 5: thresholds -----------------------------------------------------
"""<ul>
<li><b>Which matrix is more useful? The test one.</b> The training matrix shows how well the model reproduces answers it was explicitly fitted to, which is almost always flattering. Only the test matrix estimates behaviour on new leaves. That said, the training matrix is still worth a look, because <em>comparing</em> the two is how you spot overfitting.</li>
<li><b>What the threshold does.</b> It moves cases between the columns. Raising it makes the model more reluctant to say "infected": true positives fall, false negatives rise, false positives fall, true negatives rise. Lowering it does the reverse. The row totals never change, because those are the true labels. So recall and precision move in opposite directions and accuracy has an optimum somewhere in between.</li>
<li><b>Prefer the neural network? Yes</b>, but not because its accuracy number is bigger. The reason is that it actually discriminates: an AUROC well above 0.5 means it genuinely ranks infected leaves above healthy ones, so it carries information about the image. It also identifies healthy leaves, which the previous model never did, and it gives you a threshold worth tuning, a meaningless notion for a model whose output is constant.</li>
<li><b>The quarantine station.</b> There, a missed infection is the expensive error, so you <em>lower</em> the threshold and call things infected on weaker evidence. That raises recall. What it costs is precision: more false alarms, more healthy consignments delayed, inspected or destroyed. Note that this is a policy decision about which error you can better afford, not a modelling decision, the model is the same either way.</li>
</ul>""",

# --- Step 6: mistakes -------------------------------------------------------
"""<ul>
<li><b>The false negatives.</b> They tend to be leaves where the symptoms are small, sparse or early-stage, or diseases that simply look close to healthy at 64 &times; 64, faint viral mottling, or a handful of tiny spots. Some are just blurry or awkwardly lit. Grouping them by disease is informative: if one category dominates, that is a concrete instruction about which data to collect next.</li>
<li><b>Moving the threshold.</b> Raise it and false negatives grow while false positives shrink; lower it and the reverse. You cannot empty both at once unless the score distributions of the two classes do not overlap at all, that is, unless the model is perfect. That overlap <em>is</em> the model's real limitation; the threshold only lets you choose where within it you want to stand.</li>
<li><b>The next 200 leaves.</b> Collect more of what it gets wrong, not more of what it already handles. Concretely: more healthy leaves, since they are the minority class and carry most of the relative error, and more of whichever disease dominates the false negatives, specifically mild and early-stage cases rather than more textbook-obvious ones. Adding easy examples of a class the model already nails buys you almost nothing.</li>
</ul>""",

# --- Step 7a: the pre-commitment question -----------------------------------
"""<p><b>The prediction to make:</b> training performance will look <em>excellent</em>, quite possibly better than before, because brightness has become a perfectly reliable clue within the training set. Test performance will be dreadful, most likely worse than guessing, because in the test set that same clue points the wrong way.</p>
<p>If your instinct was "it should be fine, the leaves themselves have not changed", that is exactly the intuition this step is designed to break. The model does not see leaves. It sees numbers, and we changed the numbers.</p>""",

# --- Step 7b: the confounded results ----------------------------------------
"""<ul>
<li><b>The performance.</b> Training accuracy is high, test accuracy is poor, and the test AUROC lands around or below 0.5. Note carefully that the model has <em>not</em> failed to learn. It has learned extremely well, it has just learned the wrong thing.</li>
<li><b>The explanation.</b> In the training data every healthy leaf was darker. Brightness is a far simpler and more reliable pattern than lesion shape or texture, so gradient descent found it first and leaned on it: "dark means healthy". There was never any pressure to look at the disease, because the shortcut already gave a perfect answer. In the test set the darkening was applied to the infected leaves instead, so that learned rule now points precisely the wrong way.</li>
<li><b>AUROC below 0.5.</b> AUROC is the probability that the model scores a randomly chosen infected leaf above a randomly chosen healthy one. 0.5 is a coin flip. <em>Below</em> 0.5 means it reliably ranks them the wrong way round, which is not an absence of information, it is real information being applied backwards. (Inverting every prediction would score above 0.5.) It arises exactly when a cue the model learned is anti-correlated with the label in the new data, which is what we engineered here.</li>
<li><b>Catching it beforehand.</b> Nothing in the loss curves could have revealed it, and that is the uncomfortable part. In a normal project the training and test sets come from the same batch of photos, so they share the same bias and both look fine. The failure only surfaced because our test set came from a different regime. The general defences: build validation sets that genuinely differ in conditions (other farms, other days, other cameras) instead of randomly splitting one batch; record metadata such as time of day, device and operator, and check whether it predicts the label; probe what the model responds to, for instance by occluding parts of the image or testing on deliberately manipulated versions; and look at the images and how they were acquired, not only at the metrics.</li>
<li><b>Mitigations.</b> Image augmentation (step 8). Standardising the acquisition protocol so lighting cannot vary with class. Per-image normalisation, so absolute brightness carries no information at all. Deliberately balancing the confounder across classes at collection time. Holding out a validation set from a different session. Best of all, fixing the data collection rather than patching it afterwards.</li>
</ul>""",

# --- Step 8: augmentation ---------------------------------------------------
"""<ul>
<li><b>How augmentation helps.</b> By randomly re-rolling brightness, contrast and mirroring on every pass, those properties stop predicting the label: the same healthy leaf is sometimes dark and sometimes bright. The shortcut becomes useless, so the only way left to reduce the loss is to rely on cues that survive the randomisation, the lesions and colour patterns of the leaf itself. As a bonus it enlarges the effective variety of the training set, which reduces plain memorisation too.</li>
<li><b>Where the improvement came from.</b> Not from new information, from <em>destroying misleading</em> information. We injected our own knowledge, namely "brightness and mirroring must not matter for this task", as a constraint on what the model is permitted to rely on. That is prior biological knowledge supplied through the data rather than through equations, which is a nice echo of the mechanistic-versus-data-driven contrast from step 0.</li>
<li><b>Is it a complete fix? No.</b> Compare it against the step 4 model: augmentation recovers a great deal, but it usually still falls short of the model trained on clean data. Augmentation makes brightness harder to exploit, but the training set is still biased, the model has less clean signal to work with, and the random perturbations add noise to every image. Repairing the data collection would always have been better than patching around it afterwards.</li>
<li><b>Rotation versus recolouring.</b> Rotating a leaf by 180&deg; is safe because an upside-down leaf is still the same leaf with the same disease: the label is unchanged, so you are teaching the model a true fact, that orientation is irrelevant. Recolouring green to brown is a disaster because colour <em>is</em> the symptom, browning, yellowing and mottling are how several of these diseases present. You would be manufacturing images whose label no longer matches their content, teaching the model that a brown leaf can be healthy. The rule of thumb: only augment with transformations that genuinely leave the label untouched.</li>
<li><b>The optional investigations.</b>
  <ul>
    <li><em>Smaller dataset or different class balance:</em> shrinking the data lowers test performance and widens the train/test gap, because small datasets are memorised faster. Pushing the balance towards extreme imbalance makes accuracy look better while AUROC and minority-class recall get worse, the step 3 lesson all over again.</li>
    <li><em>Another species:</em> almost certainly poor without retraining. The model has only ever seen tomato leaf shapes and tomato symptoms, and other species differ in both morphology and disease repertoire. You would measure performance on a labelled sample of the new species first, and most likely fine-tune on it rather than assume transfer.</li>
    <li><em>Other metrics:</em> precision, recall and F1; balanced accuracy; Cohen's kappa; Matthews correlation coefficient; the area under the <em>precision-recall</em> curve, which is more informative than ROC AUC under strong class imbalance; recall broken down per disease; and calibration measures if you intend to treat the scores as probabilities.</li>
  </ul>
</li>
</ul>""",
]


# One entry per <div class="callout q"> block in docs/go_enrichment.html, in page order.
# Written at the reading level of the page itself: this practical is for students
# with little biology or modelling background.
GO_ENRICHMENT_ANSWERS = [

# --- Step 1: the clusters and what they leave open ---------------------------
"""<ul>
<li><b>What the dataset is.</b> It is RNA sequencing of <em>Arabidopsis</em> leaves from Howard et al. (2013), the same data you clustered in the transcriptomics practical. Plants were infiltrated with a liquid and sampled 1, 6 and 12 hours later. The three treatments are <b>MOCK</b> (buffer only, no bacteria at all), <b>AVR</b> (a bacterial strain the plant recognises, so it mounts a successful defence) and <b>VIR</b> (a strain the plant fails to stop, so it gets sick). Each number is how active a gene is relative to its own average across the nine samples, which is why the values are centred on zero. The reason for three treatments is that they let you separate three different things: what happens because the plant was handled and infiltrated at all (MOCK), what happens because bacteria are present (AVR and VIR), and what happens specifically when the plant wins (AVR).</li>
<li><b>How many clusters.</b> Around four is a reasonable choice at the default settings: you get four groups whose average profiles have clearly different shapes. Push the slider higher and you mostly split the two large groups into smaller versions of the same shape, which adds detail without adding meaning. Changing the settings does move the boundaries: <em>single</em> linkage tends to peel off a few tiny clusters and leave one enormous one, <em>complete</em> gives more evenly sized groups, and <em>euclidean</em> distance groups genes by how strongly they change rather than by the shape of the change. There is <b>no single right answer</b>. Clustering is exploratory, and the honest test is whether the groups you get are interpretable and whether the conclusions you draw in step 3 survive a change of settings. If a story only appears at one specific value of k, it is not a story.</li>
<li><b>The promising cluster.</b> At the default settings (k = 4, correlation, average) that is <b>C2</b>, the small cluster of 18 genes. What makes it convincing is not that it goes up, but <em>when and where</em> it goes up: it rises in AVR at 6 and 12 hours, much less in VIR, and not at all in MOCK. Because the MOCK plants went through the same handling but received no bacteria, a pattern that is absent in MOCK cannot be explained by the procedure. And because it is strongest in AVR, the treatment in which the plant successfully defends itself, it is a candidate for the resistance response specifically. Compare that with C1, which is much larger and rises sharply at 1 hour, but rises just as much in MOCK, so it cannot be about the bacteria. If you changed the settings your cluster numbers will differ; the pattern to look for is the one that separates infected plants from MOCK.</li>
</ul>""",

# --- Step 2: one gene at a time ----------------------------------------------
"""<ul>
<li><b>What your gene does, and specificity.</b> The exact terms depend on the gene you picked, but the shape of the answer is always the same: a handful of terms, and they are <b>not</b> equally specific. You will see very broad ones such as <em>response to stress</em>, <em>cellular process</em> or <em>metabolic process</em> alongside narrow ones such as <em>photosynthesis, light harvesting</em> or <em>response to high light intensity</em>. The graph shows this directly: general terms sit at the top, specific ones at the bottom, and a gene inherits every term above the one it is labelled with. This matters because a broad term is close to useless on its own. Saying a gene is involved in "a metabolic process" barely narrows anything down, since a large share of all genes carry that label.</li>
<li><b>The disadvantage of one gene at a time.</b> Several, and they compound. It does not scale: nobody is going to look up 214 genes by hand. It is biased by you: because <em>you</em> choose which gene to inspect, you can unconsciously pick the one that fits the story you already had, and in a list of 200 genes there is almost always one that fits any story. It tells you nothing about the group: one photosynthesis gene in a cluster is not evidence that the cluster is about photosynthesis. And the annotations themselves are incomplete and uneven, 109 of these 500 genes have no biological-process annotation at all, and heavily studied genes carry far more terms than obscure ones simply because more people have looked at them.</li>
<li><b>Do all the terms make sense? Usually not, and that is the point.</b> You will find terms such as <em>response to cold</em>, <em>response to light intensity</em> or <em>response to salt</em> on genes in a bacterial infection experiment. That is not an error in the database. A GO term records what a gene has been <em>observed</em> to do in some experiment somewhere, not what it is doing in yours. Many stress responses share the same machinery, so the same gene turns up in cold, light and pathogen studies alike. Also check the "how we know" column: some annotations were measured in the lab, others were predicted by a computer from similarity to another gene, and those two deserve different amounts of trust.</li>
</ul>""",

# --- Step 3: the enrichment, and what to do with it ---------------------------
"""<ul>
<li><b>Do the terms match the profile?</b> At the default settings they line up well, which is the reassuring case. <b>C1</b> rises at 1 hour and falls afterwards, and comes back enriched for <em>photosynthesis</em>, <em>photosynthesis light reaction</em> and <em>response to high light intensity</em>: processes that track light and time of day rather than infection. <b>C2</b> rises in AVR at 6 and 12 hours and comes back enriched for <em>response to hypoxia</em> and <em>regulation of programmed cell death</em>. Programmed cell death is exactly what a plant does when it recognises a pathogen, it kills its own cells around the infection site to stop the bacteria spreading, so the profile and the terms tell the same story. <b>C4</b> rises late in every treatment and is enriched for <em>translation</em> and <em>protein metabolic process</em>, consistent with cells stepping up protein production. Where the profile and the terms agree you have something worth testing. Where they disagree, be suspicious of the clustering, of the annotations, or of both.</li>
<li><b>The goal, in one sentence.</b> To find out which biological processes are over-represented in a group of genes, so that a list of gene names turns into a statement about biology that you can act on.</li>
<li><b>A cluster with nothing enriched.</b> At the default settings C3 is one. This is not a failure of the method, and it is not a bug. It means these genes move together but share no <em>known</em> process, and there are several honest reasons for that: the cluster may be too small for any count to reach significance, most of its genes may be unannotated, the genes may be co-regulated for a reason nobody has written down yet, or the cluster may be a technical artefact rather than biology, C3 is driven almost entirely by a single sample being unusual. An empty result is genuinely useful information: it is a warning not to build a story on that cluster, and it is exactly the kind of result you would never notice if you had gone looking through the gene list by hand until you found something that fitted.</li>
<li><b>Designing the follow-up.</b> The useful move is to stop measuring RNA and start measuring the process the terms point at. If your cluster is enriched for programmed cell death and hypoxia in the AVR treatment, then measure cell death itself, for example by staining leaves or measuring ion leakage, in AVR, VIR and MOCK plants at 6 and 12 hours. Take two or three of the genes carrying those terms, obtain knockout or overexpression lines for them, repeat the infection, and see whether the plant's resistance actually changes, that is the step that turns a correlation into a cause. Sample more densely between 1 and 6 hours, since that is the window in which the response appears to start and you currently have no measurements there. And since the goal is resilient crops rather than <em>Arabidopsis</em>, look up the orthologues of those genes in a crop species and check whether they behave the same way, which is what the next practical does. The general pattern: <em>the enriched term tells you which assay to run, the genes carrying it tell you which plant lines to build, and the expression profile tells you when to sample.</em></li>
</ul>""",

# --- Step 4: the 1 hour puzzle -----------------------------------------------
"""<ul>
<li><b>What is happening at 1 hour.</b> Look at cluster C1 carefully. It shoots up at 1 hour in AVR, in VIR <em>and in MOCK</em>. The MOCK plants received buffer with no bacteria in it at all, so whatever is driving this cannot be the pathogen. It has to be the part of the procedure that all three groups share: the plants were handled, infiltrated with liquid, wounded slightly, and sampled at one particular moment of the day. The enriched terms confirm it, they are <em>photosynthesis</em>, <em>light reaction</em>, <em>response to high light intensity</em> and <em>response to cold</em>, all things that change when a plant is moved and disturbed, and none of them anything to do with bacteria.</li>
<li><b>Why that matters more than it looks.</b> This is the largest cluster in the dataset, so a large part of the strongest signal in the whole experiment is the experiment happening to the plant rather than the infection. GO enrichment did not explain the infection here; it exposed a confounder. Two practical consequences follow. First, always compare infected against MOCK at the same time point, never against an earlier time point, because time itself changes the plant. Second, 1 hour is the worst place in this dataset to go looking for defence genes, since the handling response drowns them out. If this feels familiar, it is the same lesson as the brightness confounder in the deep-learning practical: the method faithfully found the strongest pattern in the data, and the strongest pattern was not the one anybody was interested in.</li>
</ul>""",
]


DL_HERO_OLD = """  <p style="text-align:left; max-width: 660px; margin: 0 auto;">By the end of this practical you will be able to:</p>
  <ul style="text-align:left; color: var(--muted); max-width: 660px; margin: 8px auto 0; padding-left: 20px;">
    <li>Discuss how and why to use train/test splits for training machine learning models</li>
    <li>Explain how overfitting of neural networks can arise, and give an example of how to combat it</li>
    <li>Reflect on neural networks' dependence on (unbiased) training data</li>
    <li>List various methods of assessing model performance and discuss their up- and downsides</li>
  </ul>
  <div class="nav"><a href="index.html">&larr; All practicals</a> &nbsp;·&nbsp; <a href="mechanistic_model.html">Mechanistic Modelling</a> &nbsp;·&nbsp; <a href="go_enrichment.html">GO Enrichment &rarr;</a></div>"""

DL_HERO_NEW = """  <p style="max-width: 660px; margin: 0 auto;">This is the answered version of the deep-learning practical. Every ❓ question is followed by a worked model answer (green box). All the interactive steps still work, so you can keep training models while you read, and the self-test quiz at the end is unchanged.</p>
  <div class="nav"><a href="deep_learning.html">&larr; Student version</a> &nbsp;·&nbsp; <a href="index.html">All practicals</a></div>"""

GO_HERO_OLD = """  <p style="text-align:left; max-width: 660px; margin: 0 auto;">By the end of this practical you will be able to:</p>
  <ul style="text-align:left; color: var(--muted); max-width: 660px; margin: 8px auto 0; padding-left: 20px;">
    <li>Describe how GO enrichment is applied in combination with clustering</li>
    <li>List how GO enrichment can be used to propose new experiments</li>
  </ul>
  <div class="nav"><a href="deep_learning.html">&larr; Deep Learning</a> &nbsp;·&nbsp; <a href="index.html">All practicals</a></div>"""

GO_HERO_NEW = """  <p style="max-width: 660px; margin: 0 auto;">This is the answered version of the GO-enrichment practical. Every ❓ question is followed by a worked model answer (green box). All the interactive steps still work, so you can keep clustering and re-running the analysis while you read, and the self-test quiz at the end is unchanged.</p>
  <div class="nav"><a href="go_enrichment.html">&larr; Student version</a> &nbsp;·&nbsp; <a href="index.html">All practicals</a></div>"""

# The answer boxes read better with a faint green fill, which the hand-built
# mechanistic_model_answers.html uses. Applied only to the generated page so the
# student page keeps its own (unused) rule.
ANSWER_CSS_OLD = """  .callout.answer { border-left-color: var(--accent); color: var(--ink); margin-top: 12px; }
  .callout.answer strong { color: var(--accent); }"""

ANSWER_CSS_NEW = """  .callout.answer { border-left-color: var(--accent); background: rgba(76,195,138,.09); color: var(--ink); margin-top: 12px; }
  .callout.answer strong { color: var(--accent); }
  .callout.answer ul { margin: 6px 0 0; }"""


PAGES = {
    "deep_learning": {
        "src": "deep_learning.html",
        "dst": "deep_learning_answers.html",
        "answers": DEEP_LEARNING_ANSWERS,
        "swaps": [
            ("<title>Deep Learning on Leaf Images</title>",
             "<title>Deep Learning on Leaf Images: Answer Model</title>"),
            ('<div class="badge">EduXR · Plant Breeding</div>',
             '<div class="badge">EduXR · Answer Model</div>'),
            ("<h1>Deep Learning on Leaf Images</h1>",
             "<h1>Deep Learning on Leaf Images: Answer Model</h1>"),
            (DL_HERO_OLD, DL_HERO_NEW),
        ],
    },
    "go_enrichment": {
        "src": "go_enrichment.html",
        "dst": "go_enrichment_answers.html",
        "answers": GO_ENRICHMENT_ANSWERS,
        "swaps": [
            ("<title>GO Enrichment Analysis</title>",
             "<title>GO Enrichment Analysis: Answer Model</title>"),
            ('<div class="badge">EduXR · Plant Breeding</div>',
             '<div class="badge">EduXR · Answer Model</div>'),
            ("<h1>GO Enrichment Analysis</h1>",
             "<h1>GO Enrichment Analysis: Answer Model</h1>"),
            (GO_HERO_OLD, GO_HERO_NEW),
            (ANSWER_CSS_OLD, ANSWER_CSS_NEW),
        ],
    },
}


def build_page(key: str) -> None:
    page = PAGES[key]
    src, dst = DOCS / page["src"], DOCS / page["dst"]
    answers = page["answers"]
    html = src.read_text(encoding="utf-8")

    blocks = list(re.finditer(r'<div class="callout q">.*?</div>', html, re.S))
    if len(blocks) != len(answers):
        raise SystemExit(
            f"docs/{page['src']} has {len(blocks)} question blocks but "
            f"{len(answers)} answers are defined. Update {key} in this script."
        )

    # Insert from the back so earlier offsets stay valid.
    for block, answer in zip(reversed(blocks), reversed(answers)):
        indent = " " * 4
        rendered = (
            f'\n{indent}<div class="callout answer">\n'
            f'{indent}  <strong>✅ Answer</strong>\n'
            + "\n".join(indent + "  " + line for line in answer.splitlines())
            + f"\n{indent}</div>"
        )
        html = html[: block.end()] + rendered + html[block.end():]

    # Header / title / navigation.
    for old, new in page["swaps"]:
        if html.count(old) != 1:
            raise SystemExit(
                f"docs/{page['src']}: expected exactly one occurrence of {old[:60]!r}, "
                f"found {html.count(old)}"
            )
        html = html.replace(old, new)

    dst.write_text(html, encoding="utf-8")
    print(f"wrote {dst} ({dst.stat().st_size/1024:.0f} KB, "
          f"{len(answers)} answer blocks inserted)")


def main() -> None:
    keys = sys.argv[1:] or list(PAGES)
    for key in keys:
        if key not in PAGES:
            raise SystemExit(f"unknown page {key!r}; known pages: {', '.join(PAGES)}")
        build_page(key)


if __name__ == "__main__":
    main()
