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


# One entry per <div class="callout q"> block in docs/translational_biology.html,
# in page order. Same reading level as the page: this practical assumes very
# little biology.
TRANSLATIONAL_BIOLOGY_ANSWERS = [

# --- Step 1: why look for the same gene twice --------------------------------
"""<ul>
<li><b>Why orthologues matter.</b> Turn it around and ask what you could do without them: almost nothing. Every result you produced this week is about <em>Arabidopsis</em>, and nobody grows <em>Arabidopsis</em>. Without a way of saying "this gene here corresponds to that gene there", a finding in the model plant stays in the model plant. The orthologue is the bridge. It tells a wheat breeder which of their 107,000 genes to look at, which one to screen their varieties for, which one to knock out or to select on. It also works in the other direction: if a gene is present and similar in many species, that is evidence it does something worth keeping, and a gene that only exists in <em>Arabidopsis</em> is a much weaker bet for a crop programme.</li>
<li><b>Which difference matters most.</b> There is no single right answer, but the two that usually come first are <b>generation time</b> and <b>how easy it is to add or remove a gene</b>. Six to eight weeks from seed to seed against five to eight months means you can run six experiments in the time a wheat researcher runs one, and genetics is a subject where you often need several rounds of crossing before you learn anything. Being able to change a gene routinely is what lets you go from "this gene is correlated with resistance" to "this gene causes resistance", which is the step that actually settles the question. The genome size and the six sets of chromosomes matter too, but mostly because they make everything else harder: with six copies of every chromosome, knocking out one copy of a gene often changes nothing at all, which is a problem you will meet again in step 6.</li>
</ul>""",

# --- Step 2: orthologue or paralogue ------------------------------------------
"""<ul>
<li><b>The difference.</b> Both are pairs of genes that descend from one single gene in some ancestor. What separates them is the event that made the two copies. If the two copies were separated because <b>the species split in two</b>, they are <b>orthologues</b>. If the two copies were made because <b>the gene was copied inside one genome</b>, they are <b>paralogues</b>. That is the entire definition, and notice that it says nothing about how similar the two sequences are. Two paralogues can be nearly identical and two orthologues can be quite different; similarity is a clue to the history, not the history itself.</li>
<li><b>Which one you want.</b> You want the <b>orthologue</b>. The reasoning is about what the two kinds of pair have been doing since they separated. Orthologues have been sitting in two different species, each doing the same job the ancestral gene did, because there is only one copy in each species and nothing else to take that job over. Paralogues sit side by side in one genome, where one copy is enough to keep the plant running, so the second copy is free to drift, pick up a new role, or stop working altogether. So the orthologue is the pair with a reason to have kept the same function, and the paralogue is the pair with a reason not to. That is a tendency rather than a law, which is what step 6 is about.</li>
<li><b>Why the order of events changes everything.</b> In "copied in both", the split happened first. Every gene on the Arabidopsis side and every gene on the rice side meet at that split when you trace them upwards, so all four cross-species pairs are orthologues, and because there are two copies on each side that is a many-to-many relationship. In "copied before the split", the copying happened first. Trace A1 and R1 upwards and you meet a species split, so they are orthologues, and the same is true of A2 and R2. But trace A1 and R2 upwards and you go past both splits before the two paths meet, at the copying event, so those two are <b>paralogues even though they are in different species</b>. The four genes at the bottom are identical in both pictures. What differs is only which event you land on when you trace two of them back, and that is what the words are defined by. This is also exactly why the tree-based method in step 3 beats simply looking for the most similar sequence: A1 and R2 might well be each other's best match, and they would still be the wrong answer.</li>
</ul>""",

# --- Step 3: the real Compara result -------------------------------------------
"""<ul>
<li><b>Does your gene have orthologues?</b> Almost certainly yes: 479 of the 500 genes have at least one in these ten species, so if you picked at random you probably got one. Which species you got depends on the gene. Cabbage is the most reliable, missing for only 38 of the 500, which makes sense because it is in the same plant family as <em>Arabidopsis</em> and separated from it most recently. Barley is the least reliable, missing for 240 of them. The 21 genes with nothing anywhere are worth a look on their own.</li>
<li><b>Why some genes have far more than others.</b> Try <code>AT1G79040</code> (PSBR) and then <code>AT3G53260</code> (PAL2) to see the extremes: 7 orthologues against 149. Several things drive the difference and they stack up. Some genes belong to <b>large families</b> that have been copied over and over in every lineage, so one <em>Arabidopsis</em> gene meets dozens of relatives in each species. Some genomes have been <b>doubled or tripled</b> more recently than others, which multiplies every gene in them at once. Some genes are so central to staying alive that every species has kept one, while others are specific to a way of life and get lost in species that do not need them. And some of the difference is not biology at all: a well-studied genome is annotated better than a poorly studied one, so it is easier to find orthologues in it.</li>
<li><b>What can go wrong with similarity alone.</b> Step 2 gave the answer: the most similar sequence is not always the orthologue. In the "copied before the split" history, A1 and R2 are paralogues, yet nothing about their sequences announces that, and if the copying event was recent they may well look more alike than the true orthologue pair. Similarity also gets misread in the other direction: two orthologues that have both changed a lot no longer look like each other's best match, so a real relationship gets missed. Similarity is evidence about history, and treating evidence as if it were the conclusion is the mistake.</li>
</ul>""",

# --- Step 4: the counts on the tree ---------------------------------------------
"""<ul>
<li><b>What the tree shows.</b> The tree is a picture of how the ten species are related to each other. Each <b>branch point</b> is a moment in the past when one ancestral species split into two, and the further left a branch point is, the longer ago that happened. It is <em>not</em> a picture of your gene: the same tree is drawn no matter which gene you select. What changes with your gene is the annotation on the right. The <b>number</b> beside a species is how many genes in that species Ensembl calls orthologues of your one <em>Arabidopsis</em> gene, and the <b>colour</b> is the relationship it reports for them.</li>
<li><b>Why the relationship differs.</b> It comes straight out of step 2, applied to the real tree. One-to-one means one copy on each side of the species split, so nothing was copied afterwards in either lineage. One-to-many means one copy stayed single in <em>Arabidopsis</em> while the other lineage copied it. Many-to-many means both lineages copied it. Compare the two contrasting genes: <code>AT1G79040</code> (PSBR) has at most two copies anywhere and is missing entirely from the grasses and moss, while <code>AT3G53260</code> (PAL2) is a member of a large family that has been copied repeatedly in every lineage, giving 54 copies in wheat alone. Same tree, same method, completely different picture, because the two genes have had completely different histories.</li>
<li><b>Why wheat has more of everything.</b> Look at the table in step 1: bread wheat has <b>six sets of chromosomes</b> where <em>Arabidopsis</em> has two. Bread wheat formed when three related grass species combined their whole genomes, so it carries three near-complete copies of a grass genome at once. Every gene is therefore present roughly three times before you even start counting gene families, and in this dataset wheat averages 8.8 orthologues per gene where every other species averages between 1.5 and 3.6. This is the single most common reason a crop gives you a one-to-many result, and it is not a quirk of wheat: potato, soybean, maize and cabbage have all been through genome doublings of their own.</li>
<li><b>Nothing found.</b> Two quite different things could be true, and they call for different responses. Either the gene <b>genuinely is not there</b>, because it was lost in that lineage or never existed before it, or the gene <b>is there and we failed to find it</b>, because the genome is incompletely sequenced, badly annotated, or the sequence has changed so much that the search no longer recognises it. Check the second one first, because it is much more common and much cheaper to check. A good test is to look at the neighbours on the tree: CAB1 has 54 orthologues in wheat and 3 in rice but <b>none in barley</b>, and it is not credible that barley alone among the grasses lost a photosynthesis gene. That is an annotation gap, not biology. When a whole branch of the tree comes back empty, that is when a real loss becomes the better explanation.</li>
</ul>""",

# --- Step 5: the alignment ------------------------------------------------------
"""<ul>
<li><b>Do you agree?</b> For most pairs, yes, and the reason is the <em>pattern</em> rather than the percentage. What convinces you is a long stretch of the protein matching almost letter for letter, running the full length of both sequences, with the mismatches scattered as single positions rather than piled up in one place. Two unrelated proteins do not do that. Around 20 to 25 percent of positions match by chance alone between any two protein sequences, so a figure near that means nothing, while 87 percent running end to end between two species that separated 150 million years ago is not something chance produces.</li>
<li><b>Why it is not perfect.</b> The two lineages have been evolving separately ever since the species split, and two different kinds of change accumulate. <b>Single differing positions</b> come from point mutations, where one letter of the DNA changed and the amino acid at that position changed with it. Most of these are harmless, which is why they survive; notice how many of them are the "similar" colour, meaning the replacement amino acid has much the same chemical character and the protein carries on working. <b>Gaps</b> come from a different kind of event: a piece of DNA was inserted into one lineage or deleted from the other, so one protein has a stretch the other does not. Gaps in the middle of a protein are relatively rare because losing a chunk of a working protein usually breaks it, and you will notice that most gaps sit near the ends.</li>
<li><b>Close against distant.</b> The closer the species, the higher the identity and the fewer the gaps. For CAB1 the cabbage orthologue comes in around 97 percent while the rice one is around 87 percent, and that ordering follows the tree in step 4 exactly: less time since the split means less time to accumulate change. This is worth noticing because it is the same logic running in reverse. We use trees to interpret sequences, and sequence differences are how the trees were built in the first place.</li>
<li><b>Conserved and variable regions.</b> The regions that match almost perfectly are almost always the parts that <em>have</em> to be right: the active site of an enzyme, the surface where the protein grips another molecule, the core that holds it folded. A mutation there breaks the protein and the plant carrying it does worse, so those changes never spread. The messy regions are the parts where the exact sequence matters less, such as flexible linkers between the working parts, or the signal at the start of the protein that says which compartment of the cell to send it to, where the general character matters but the individual letters do not. So an alignment is not only evidence about ancestry, it is also a rough map of which parts of a protein matter, which is genuinely useful if you are deciding where to aim a mutation.</li>
</ul>""",

# --- Step 6: doing it for real, and the caveat -----------------------------------
"""<ul>
<li><b>The literature check.</b> What you find depends entirely on the gene, and the honest outcome for most of them is that <b>nothing has been published</b> on the crop orthologue at all. That is itself the answer to why this practical exists. Where you do find something, the usual pattern is a broad match with a specific mismatch: the crop gene turns out to be involved in the same general process, but switched on at a different time, in a different tissue, or in response to a different stress. Treat a match as encouragement rather than proof, and treat a mismatch as interesting rather than as an error, because a gene that has been repurposed in a crop is worth understanding. Also check what kind of evidence you have found: a paper that measured the gene is worth far more than a database entry that inferred its function from the <em>Arabidopsis</em> gene, which would just be your own assumption handed back to you.</li>
<li><b>Knocking out one of three copies.</b> Most likely <b>nothing visible happens</b>, and your experiment tells you nothing. The other two copies are still there and still doing the job, so the plant carries on as before. This is called redundancy and it is the standard frustration of working in wheat: the six sets of chromosomes that make the genome big also mean that single knockouts are usually silent. The consequences are practical. You may need to knock out all three copies together to see any effect at all, which is far more work. A negative result from a single knockout is close to uninformative, so you should not publish one as evidence that the gene does not matter. And if you have to choose, it can be smarter to work in a crop with fewer genome copies first, such as barley or rice, and move to wheat once you know what you are looking for.</li>
<li><b>The experiment.</b> A reasonable answer chains the whole week together and stays concrete. For example: the clustering put a group of genes together that rise in the AVR treatment at 6 and 12 hours but not in MOCK, GO enrichment says that group is enriched for programmed cell death, and Ensembl gives orthologues of the three strongest of those genes in barley, which is a real crop with only two sets of chromosomes. So: obtain or generate knockout lines for those three barley orthologues, infect them and wild-type barley with a comparable pathogen, and measure both the disease itself and the process the GO term pointed at, for instance by staining for cell death, at the same time points the original experiment used. Include a mock-inoculated control at every time point, for the reason step 4 of the GO practical made painfully clear. The point to get across is the shape of the argument: <em>clustering says which genes move together, GO says what they are probably doing, the orthologue says where to test it, and only the experiment says whether it is true.</em></li>
</ul>""",
]


# One entry per <div class="callout q"> block in docs/mechanistic_model.html, in
# page order. Lifted from the hand-built answer page this script replaced.
MECHANISTIC_MODEL_ANSWERS = [

# --- Step 1: reading the network --------------------------------------------
"""<ul>
<li><b>Drought resistance</b> runs through the ABA branch: drought raises the hormone <b>ABA</b>, which drives <b>ABF2 → ANAC019</b> and <b>GENEC</b>. <b>White-fly (biotic) resistance</b> is the salicylic-acid pathway <b>ICS1 → SA → BGL2</b>; <b>BGL2</b> is the defence marker we read out. The clever bit the paper highlights is the <em>link</em> between them: the drought (ABA) branch represses the defence (ICS1) branch.</li>
<li><b>Arrows:</b> a green solid arrow means the source gene <b>activates</b> (switches on) its target; a red dashed arrow means it <b>represses</b> (switches off) its target.</li>
<li><b>No drought:</b> with drought = 0, ABA stays low, so ABF2, ANAC019 and GENEC stay low. Nothing is repressing ICS1, so ICS1 is expressed → SA is produced → BGL2 is high. The resting plant is <b>fully defended</b>.</li>
</ul>""",

# --- Step 2: the resting plant ------------------------------------------------
"""<ul>
<li><b>Yes, it is resistant.</b> Because there is no drought, ABA (and therefore ANAC019 and GENEC) stay near zero, so nothing represses ICS1. ICS1 climbs to a high steady level, driving SA and then BGL2 up. High BGL2 = active salicylic-acid defence, so the plant can fend off white flies.</li>
<li><b>Prediction for a brief pulse:</b> the ABA branch will spike briefly, nudging the repressors up for a moment, so defence may dip a little, but because the pulse is short we would expect it to recover quickly. (Confirm this in the next step.)</li>
</ul>""",

# --- Step 3: the brief pulse --------------------------------------------------
"""<ul>
<li><b>What happens:</b> the pulse briefly raises ABA, which activates ABF2 → ANAC019 and GENEC, so you see those upstream genes spike. They momentarily repress ICS1, so ICS1/SA/BGL2 dip a little. But the pulse is over almost immediately, ABA decays, the repression lifts, and the defence pathway climbs back to its resting level. Net effect: a small, temporary dip and full recovery.</li>
<li><b>Why it makes sense:</b> brief dry spells are normal and not life-threatening. It would be wasteful (and risky) to throw away biotic defence every time it gets dry for an hour. Requiring a <em>sustained</em> signal before switching off defence lets the plant ignore short, noisy fluctuations and only reallocate resources when drought is genuinely serious, which is a filter against over-reacting.</li>
<li><b>Prediction:</b> a longer drought should suppress ICS1/SA/BGL2 more strongly and for longer, so white-fly resistance should drop noticeably. (Test it next.)</li>
</ul>""",

# --- Step 4: the sustained drought --------------------------------------------
"""<ul>
<li><b>Why it differs:</b> a sustained drought holds ABA, and therefore ANAC019 and GENEC, high for a long time. Their combined repression keeps ICS1 switched off long enough for ICS1, SA and BGL2 to actually decay away. So defence collapses for the duration of the drought, and only recovers once the drought ends. The key variable is not <em>whether</em> the repressors turn on, but for <em>how long</em>.</li>
<li><b>Does it reflect biology?</b> Qualitatively yes: it captures a real, documented trade-off (drought signalling suppressing salicylic-acid defence) and the sensible logic that only prolonged stress flips the switch. But it is a heavy simplification: only a handful of genes, deterministic with no noise, fixed Hill parameters and arbitrary interaction strengths, no feedback loops, no tissue or spatial detail, and no fitness cost attached. Treat it as a caricature that is useful for reasoning, not as ground truth.</li>
<li><b>Which genes to knock out:</b> to protect defence, target the repressors of ICS1, namely <b>ANAC019</b> and/or <b>GENEC</b>, or cut the signal at its source by knocking out <b>ABA</b>. (Test in the next step.)</li>
</ul>""",

# --- Step 5: knockouts ---------------------------------------------------------
"""<ul>
<li><b>Best knockouts:</b> silence the two repressors of ICS1, that is <b>ANAC019 and GENEC together</b>. Then even a long drought cannot switch ICS1 off, so SA and BGL2 stay high and the plant keeps its white-fly defence throughout. Knocking out <b>ABA</b> also works, by removing the drought signal before it ever reaches the repressors. Note the trap: knocking out ICS1, SA or BGL2 themselves <em>destroys</em> defence, the opposite of what you want.</li>
<li><b>Realistic?</b> Not straightforwardly. Defence is metabolically costly, so a plant that can never dial it down may waste resources it needs to survive drought, hurting growth or yield, a cost this model does not include. Transcription factors like ANAC019 are also pleiotropic (they regulate many other genes), so knocking them out likely has side effects far beyond this little network. The model also ignores off-target effects, environment and gene-gene redundancy.</li>
<li><b>Next experiment and communication:</b> create the knockout line (for example by CRISPR, or use an existing mutant), then measure both traits under a real drought: white-fly resistance and SA/BGL2 levels <em>and</em> drought survival, growth and yield, in controlled and ideally field conditions. Present the result to a biologist as a <em>model-generated hypothesis</em> with explicit caveats: "the model predicts X; here is the trade-off it ignores; here is the experiment that would confirm or refute it."</li>
</ul>""",
]


# One entry per <div class="callout q"> block in docs/marker_assisted_selection.html,
# in page order. This is the first practical of the course, so the answers spell
# out the reasoning rather than assuming any modelling background.
MARKER_ASSISTED_SELECTION_ANSWERS = [

# --- Step 1: reading the population table -------------------------------------
"""<ul>
<li><b>Rows and columns.</b> Each <b>row</b> is one individual plant in the population. Each of the columns labelled M0 to M49 is one <b>marker</b>, that is one fixed position in the DNA, and the 0 or 1 in the cell says which of the two variants that particular plant carries at that particular position. The last two columns are different in kind: they are not DNA at all, they are the two <b>phenotypes</b> we measured on the plant, its salt resistance and its yield. That mix is the whole point of the table. Everything that follows comes from asking whether any of the 0/1 columns lines up with one of the two measured columns.</li>
</ul>""",

# --- Step 2: the Manhattan plot ------------------------------------------------
"""<ul>
<li><b>What the plot shows, and why diversity matters.</b> Each bar is one marker, and its height is <code>-log10(p)</code> from a separate statistical test asking whether plants carrying a 1 at that marker have a different salt resistance from plants carrying a 0. A tall bar means that difference is unlikely to be a fluke. The population has to be genetically diverse because the test is a <em>comparison</em>: if every plant carried a 1 at a marker, there would be no 0 group to compare against, the test would have nothing to work with, and the bar would be flat. No variation means no information, however many plants you measure.</li>
<li><b>The promising marker.</b> It is <b>Marker 12</b>, the single bar that climbs far above all the others and above the significance line. Click it and the scatter plot below separates cleanly into two horizontal bands: the plants with a 0 sit around 30 percent salt resistance, the plants with a 1 sit around 80. The two groups barely overlap, which is exactly what a marker with a large effect looks like. Try clicking a few other bars for contrast; there the two groups sit on top of each other and you cannot tell them apart.</li>
<li><b>Growing the population.</b> At 3 plants the plot is noise: several bars look tall, and Marker 12 is not obviously special. As you drag towards 100, the bar at Marker 12 climbs higher and higher while every other bar stays low and flat. Nothing about the biology changed, only the amount of evidence. With few plants a coincidence is easy, so no marker can be distinguished from chance; with many plants a real difference accumulates evidence and the coincidences do not. This is why a GWAS needs hundreds or thousands of individuals, and why a peak found in a small population should be treated with suspicion.</li>
</ul>""",

# --- Step 4: running the backcrossing programme ---------------------------------
"""<ul>
<li><b>What to cross with, and what to select on.</b> Cross back to the <b>elite (agricultural) variety</b> every round, and keep the offspring with <b>Marker 12 = 1</b>. The reasoning has two halves that pull in opposite directions. Each cross to the elite parent replaces roughly half of the remaining donor genome with elite genome, so after a few rounds the plants are almost entirely elite and the yield climbs back towards the elite level. But that same random halving would also throw away the salt-resistance allele about half the time, which is why you filter: selecting on Marker 12 = 1 protects the one piece of donor DNA you actually wanted. Crossing back to the donor, or to the population itself, makes no progress towards elite yield, and selecting on any other marker does nothing to protect resistance.</li>
<li><b>Why select on a marker rather than on the plant.</b> You can read a marker from a leaf sample of a seedling, days after germination, instead of growing the plant to maturity, and you never have to expose it to salt to find out whether it is resistant. That is faster, cheaper and non-destructive. It is also more reliable: a measured phenotype is the sum of the genetics and the environment, so a resistant plant in a bad spot can look sensitive and mislead you, whereas the marker reads the genetics directly. In this simulation you can see the environment as the noise term that scatters the points in the step 2 scatter plot.</li>
<li><b>Reading success off the graph.</b> Success is <em>both</em> lines ending up where you want them, not one. Salt resistance should start high and stay high, near 80, round after round. Yield should start low, because the donor is a poor yielder, and climb towards the dashed elite target line as the elite genome is recovered. If salt resistance collapses you lost the allele; if yield flattens out well below the dashed line you are not backcrossing to the elite parent. Watch the round cards too: they tell you how many of the 20 offspring passed the filter each round.</li>
<li><b>Crossing without a filter.</b> Salt resistance drifts down, usually in steps, and often collapses to around 30 within a few rounds. The reason is chance rather than selection: each offspring inherits each marker from one parent or the other at random, so an offspring of a resistant plant crossed with the elite has only a 50 percent chance of receiving the resistance allele. With no filter, the fraction of carriers halves on average every round, and once it reaches zero it cannot come back, because nothing in the remaining population carries the allele any more. This is the single most important thing the simulation shows: the cross alone does not preserve a trait, the <em>selection</em> does.</li>
</ul>""",

# --- Step 5: the yield Manhattan plot -------------------------------------------
"""<ul>
<li><b>Comparing the two plots.</b> They look completely different, and the difference does not go away as the population grows. The salt plot develops one dominant spike at Marker 12 that keeps climbing. The yield plot instead develops a whole cluster of modest bars, none of which dominates, and many of them stay near or below the significance line even at 100 plants. Adding plants sharpens the picture but never produces a single obvious winner.</li>
<li><b>One marker or many.</b> Many. The evidence is in the shape of the plot. No single bar towers over the rest the way Marker 12 does. That pattern is what a <b>polygenic</b> trait looks like, one where many genes each contribute a small amount to the phenotype, and it is the normal situation for complex traits like yield, height or flowering time.</li>
<li><b>What that means for marker assisted selection.</b> It largely breaks the method. Selecting on any one of those markers gains you only a sliver of the trait, and stacking them one at a time would take an impractical number of crossing rounds, with each round risking the loss of what you gained before. Worse, a marker with a small effect is hard to distinguish from a false positive, so you may spend rounds selecting on something that does nothing. The method needs a single marker carrying most of the effect, and yield does not have one. What you need instead is a way to use <em>all</em> the markers at once, adding up their small contributions into one number per plant, which is exactly what genomic selection does.</li>
</ul>""",
]


# One entry per <div class="callout q"> block in docs/genomic_selection.html, in
# page order. Follows directly on from the marker assisted selection practical.
GENOMIC_SELECTION_ANSWERS = [

# --- Step 1: what a prediction would need --------------------------------------
"""<ul>
<li><b>What you would need first.</b> You would need to know <em>how much each marker contributes to yield, and in which direction</em>. A new plant's DNA is just a row of 0s and 1s; on its own that tells you nothing. It only becomes a prediction once you can attach a number to each marker and add up the contributions. And the only way to learn those numbers is from a population like this one, where you know the markers <em>and</em> the measured yield for the same plants, so you can work out which marker values tend to travel with high yield. Note what this implies: you can never predict a phenotype from DNA alone. Somebody has to have grown and measured a training population first.</li>
</ul>""",

# --- Step 2: confirming the trait is polygenic ----------------------------------
"""<ul>
<li><b>Comparing the two plots.</b> The salt-resistance plot had one tall spike at Marker 12 that rose further and further above every other bar as the population grew. This one never produces that spike. Instead a group of markers at the left each reach a modest height, a few others dip in the opposite direction, and the rest stay flat. Growing the population makes the pattern cleaner but does not turn any bar into a winner, so this is not a problem you can solve by collecting more plants.</li>
<li><b>Why marker assisted selection fails here, and the biology behind it.</b> Marker assisted selection needs a single marker that carries most of the effect, so that checking one position in the DNA tells you most of what you need to know. Here no marker does. Selecting on the tallest bar would capture only a small slice of the trait, and stacking the useful markers one at a time would take far too many crossing rounds. The underlying biology is that yield is not one thing. It is the end result of photosynthesis, root growth, water and nutrient uptake, flowering time, stress tolerance and much else, each controlled by its own genes. Traits built out of many small contributions are called <b>polygenic</b>, and most agriculturally important traits are polygenic. Salt resistance in the previous practical was the unusual case, not the typical one.</li>
</ul>""",

# --- Step 4: reading the learned weights ----------------------------------------
"""<ul>
<li><b>What one bar tells you.</b> The <b>direction</b> says which way that marker pushes: a bar above zero means plants carrying a 1 there tend to yield <em>more</em>, a bar below zero means they tend to yield <em>less</em>. The <b>height</b> says how big that push is, in the units of the trait. So a bar of +2 is the model saying "carrying a 1 at this marker is worth about 2 extra units of yield, holding the other markers fixed". A bar near zero means the model found no consistent effect for that marker.</li>
<li><b>Which markers matter.</b> Once the model has enough training plants, the picture is clear: a block of about ten markers at the very left of the plot (markers 0 to 9) all stand well above zero, and a small block around markers 20 to 22 sits below zero. Those are the markers that raise and lower yield respectively. Everything else hovers around zero. Train on only 5 or 10 plants and this structure is buried in noise; drag the slider up and watch the real pattern emerge from the clutter. That is worth doing, because it shows you what "the model needs enough data" actually looks like.</li>
<li><b>How this differs from a Manhattan plot.</b> This is the key conceptual point of the practical, and it is easy to slide past. The two plots look similar but the bar heights mean different things. In a Manhattan plot the height is <b>evidence</b>: how confident we are that a marker is associated with the trait at all. It is always positive, it says nothing about the direction of the effect, and it grows without limit as you add more plants, because more data means more certainty. Here the height is an <b>effect size</b>: how much yield that marker is worth, with a sign. It does not grow as you add plants, it converges on the true value. Another way to put it: a Manhattan plot answers "is this marker involved?", one marker at a time; the weights plot answers "how much is this marker worth?", for all markers at once and taking the others into account.</li>
</ul>""",

# --- Step 5: predicted versus actual ---------------------------------------------
"""<ul>
<li><b>Why the predictions are not perfect.</b> Several reasons stack up, and none of them can be trained away. The biggest is that the phenotype is not purely genetic: two plants with identical DNA still yield differently because of soil, water, light and chance, and no model reading only DNA can predict that part. On top of that the model has seen a limited number of training plants, so its 50 weights are estimates rather than truths; the markers we measured may not be the causal variants themselves, only positions that happen to sit near them; and the model assumes contributions simply add up, whereas real genes interact. The scatter around the diagonal is the honest visual summary of all of that, and a model that sat exactly on the line would be a warning sign, not a triumph.</li>
<li><b>The unseen plants, and why they matter.</b> The yellow points should sit around the diagonal much like the blue ones, so yes, the model still works on plants it never saw. That is the result you actually care about. The blue training points are not evidence of anything, because the model was fitted to them: a model can score well there simply by memorising, which teaches you nothing about the next plant. Only the held-out points estimate what happens when you use the model for real, on seedlings whose yield nobody has measured, which is the entire purpose of the exercise. Set the training size very low and watch the two groups come apart: the model still tracks the blue points reasonably while the yellow ones scatter badly. That gap is overfitting, and testing on unseen plants is how you catch it.</li>
</ul>""",

# --- Step 6: the breeding cycles ---------------------------------------------------
"""<ul>
<li><b>Reading success off the graph.</b> Compare the two lines, not the green one on its own. Both programmes start from the same population, so any gap between them is caused by the selection and nothing else. The green line should climb steadily cycle after cycle while the blue control wanders roughly sideways, drifting up or down by chance. If the green line is not pulling away from the blue one, the selection is not doing anything, whatever the absolute yield happens to be. This is why the control matters: without it you could mistake ordinary variation between generations for progress.</li>
<li><b>Passing the elite line.</b> The dashed line is the best variety farmers currently grow, so crossing it means the breeding programme has produced something better than today's best. It is possible because the elite variety is good but not perfect: it carries most of the favourable alleles but is missing one, and no single existing plant happens to combine every favourable allele at once. The variation to do better is already present in the population, scattered across different individuals. Selection does not create anything new; it <b>assembles</b> favourable alleles that were already there into one plant. That is also why progress eventually slows: once the favourable alleles are all fixed, there is nothing left to assemble.</li>
<li><b>Would it work for a single-marker trait?</b> Yes, it would work, but it would be the wrong tool. The model would simply learn one large weight on Marker 12 and near-zero weights everywhere else, and selecting on the predicted value would amount to selecting on Marker 12. You would get the right answer by an expensive route: genotyping every marker and fitting a model in order to reproduce what a single marker test tells you directly and far more cheaply. There is also a real cost to the detour, since with a small training population the model may spread the effect across nearby markers or attach weight to markers that are only there by chance. The rule of thumb: use marker assisted selection when one marker explains most of the trait, and genomic selection when no single marker does.</li>
<li><b>Limitations.</b> The most important one is that the model is only valid for the population it was trained on. Its weights are not statements about biology, they are statements about which markers happened to travel with the causal genes <em>in that population</em>. Apply it to unrelated material and those associations may be weaker, absent or reversed, and the predictions degrade badly. The same applies across environments: a model trained on yield in one climate can rank plants wrongly in another, because the genes that matter are not the same ones. Then there are the slower problems. The associations decay over generations as recombination separates markers from the genes they were standing in for, so the model has to be retrained on new measured plants every so often. Selecting hard on predictions shrinks genetic variation each cycle, so gains get smaller and the population becomes inbred, and a model that only knows about yield will happily drive yield up while quietly discarding disease resistance or drought tolerance that nobody told it to protect.</li>
</ul>""",
]


# One entry per <div class="callout q"> block in docs/transcriptomics_clustering.html,
# in page order. The specific numbers below were checked against the embedded
# dataset, so they will match what students actually see on the page.
TRANSCRIPTOMICS_CLUSTERING_ANSWERS = [

# --- Step 1: the experimental design ---------------------------------------------
"""<ul>
<li><b>The three treatments and why compare them.</b> <b>MOCK</b> is the control: the plant is infiltrated with liquid but receives no bacteria at all. <b>AVR</b> is infection with an avirulent strain, one the plant recognises and successfully fights off. <b>VIR</b> is infection with a virulent strain, one that evades the plant's defences and causes disease. Having all three lets you separate three different things that would otherwise be tangled together. Comparing either infection against MOCK tells you what the bacteria did, as opposed to what the handling and infiltration did. Comparing AVR against VIR tells you what a <em>successful</em> defence looks like, as opposed to an unsuccessful one. Without MOCK you could not tell a defence response from a wounding response, and without the AVR/VIR pair you could not tell resistance from mere infection.</li>
<li><b>Why several time points.</b> Because a defence response is a sequence of events, not a state. The plant has to detect the bacteria, pass the signal on, switch on transcription factors, and only then produce defence proteins, and those stages happen minutes to hours apart. A single snapshot would catch one frame of that and you would have no way of knowing which. Several time points let you see the order in which genes come on, which is the closest this kind of data gets to telling you what causes what, and they also protect you from picking the wrong moment: sample too early and nothing has happened yet, too late and the response is already over.</li>
</ul>""",

# --- Step 2: reading the clustered heatmap ------------------------------------------
"""<ul>
<li><b>Rows, columns and colours.</b> Each <b>row</b> is one of the 500 genes and each <b>column</b> is one of the 9 samples, one treatment at one time point. The <b>colour</b> of a cell is that gene's z-scored expression in that sample: red means the gene is more active than its own average across the nine samples, blue means less active, and pale means about average. Because the scores are computed per gene, colours are only comparable <em>along</em> a row. A red cell does not mean the gene is highly expressed in absolute terms, only that it is high for that gene.</li>
<li><b>Is AVR closer to VIR or to MOCK? To VIR.</b> The dendrogram shows it twice over. At 1 hour, AVR_1 and VIR_1 merge with each other first, at a very short distance, and MOCK_1 only joins them afterwards. Among the later samples the same thing happens one level up: the two AVR samples form a pair, the two VIR samples form a pair, and those two pairs join <em>each other</em> before the MOCK pair joins them. Biologically this makes sense once you stop reading the names. Avirulent and virulent sound like opposites, but from the plant's point of view both mean bacteria are present: it detects them, mounts an attack, diverts resources away from growth. MOCK is the odd one out, because nothing is attacking it at all. The difference between winning and losing turns out to be smaller than the difference between being attacked and not being attacked.</li>
<li><b>Treatment or time point? Both, at different levels, and this is the interesting part.</b> The very first split in the tree separates the three 1-hour samples from all six later ones, so at the coarsest level <b>time</b> dominates. But look inside the later group and the structure is entirely by <b>treatment</b>: AVR pairs with AVR, VIR with VIR, MOCK with MOCK, across time points. So time is the bigger effect overall while treatment is what organises the samples once the 1-hour block is set aside. The reason the 1-hour samples are so distinctive is worth pausing on: all three of them cluster together, including MOCK, so whatever makes them special cannot be the bacteria. It is the handling, wounding and infiltration that every plant went through, plus the time of day. The practical consequence is that you should always compare an infected sample against MOCK <em>at the same time point</em>, never against an earlier one.</li>
<li><b>The clustering in your own words.</b> Something along these lines: the algorithm knew nothing about treatments or time points, it only saw 9 columns of numbers, and yet it recovered the design of the experiment. It put the earliest samples together because they share a strong handling response, and then grouped the rest by treatment, with the two infected treatments closer to each other than either is to the uninfected control. That is a genuinely useful result, and it doubles as a sanity check: if replicates or matching conditions had <em>not</em> grouped together, you would suspect a technical problem such as a mislabelled sample or a batch effect long before you started interpreting individual genes.</li>
</ul>""",

# --- Step 3: choosing a clustering --------------------------------------------------
"""<ul>
<li><b>How many clusters, and is there a right answer? No, there is not, and the honest answer is a judgement.</b> Four is a defensible choice at the default settings, because it gives four groups whose average profiles have visibly different shapes: one large group that peaks at 1 hour in every treatment, one large group that rises late, a small group of about 18 genes that rises specifically in AVR at 6 and 12 hours, and a small group of about 21 genes driven almost entirely by one unusual sample. Push the slider higher and you mostly cut the two big groups into smaller pieces with the same shape, which adds detail without adding meaning. The real test is not statistical, it is whether the groups you get are interpretable and whether your conclusions survive a change of settings. If a story only appears at one particular value of k, it is not a story.</li>
<li><b>What changes when you switch the settings.</b> <b>Linkage</b> matters enormously here. <em>Average</em> and <em>complete</em> both give sensibly sized groups, with complete producing slightly more even ones, because it refuses to merge two clusters unless even their most distant members are close, which keeps clusters compact. <em>Single</em> linkage collapses: at k = 4 it hands you one cluster of 497 genes and three single-gene clusters. That is the classic failure called <b>chaining</b>. Single linkage merges two clusters as soon as their <em>closest</em> members are close, so one gene sitting between two groups is enough to fuse them, and the process snowballs until almost everything is in one clump, leaving only true outliers behind. <b>Distance</b>, by contrast, barely matters on this dataset, which is a surprise worth explaining. The values were z-scored per gene, so every gene is already on the same scale, and for vectors like that Euclidean distance is just a rescaling of correlation distance. The two give nearly identical trees here. On raw, unscaled expression they would differ a great deal, because Euclidean would separate genes by how strongly they are expressed while correlation would group genes by the shape of their response regardless of level.</li>
<li><b>Which cluster to follow up, and how to decide.</b> The small cluster of about 18 genes that rises in AVR at 6 and 12 hours is the best candidate. The criterion is not that it changes a lot, but that it changes <em>where the biology is interesting</em>: it is high in the treatment where the plant successfully defends itself, much weaker in VIR, and flat in MOCK, so the handling cannot explain it. Compare that with the large cluster that peaks at 1 hour, which rises just as much in MOCK and therefore cannot be about the bacteria at all, and with the small cluster driven by a single odd sample, which looks striking but is most likely a technical artefact. So the decision rule is: prefer a cluster whose pattern lines up with a comparison you deliberately built into the experiment, be suspicious of one that is driven by a single sample, and check that the genes in it are more than a handful before you build a story on them. The next practical takes exactly this cluster and asks what the genes in it actually do.</li>
</ul>""",

# --- Step 4: the correlation network ---------------------------------------------------
"""<ul>
<li><b>What the threshold does.</b> It sets how strong a correlation has to be before two genes are joined, so it controls how much of the web survives. At 0.85 almost every gene is connected to something, roughly 15,000 edges among 489 genes, and the result is a hairball in which nothing stands out. Raise it to 0.95 and you are down to about 2,900 edges among 378 genes, with distinct knots becoming visible. At 0.99 only about 156 edges and 118 genes remain, leaving a few small, very tightly linked groups. Note that the threshold is a choice you make, not something the data tells you: too low and everything looks connected to everything, too high and you discard real relationships. Move it up and down and check whether the group you are interested in holds together across a range of values, because a knot that only exists at one threshold is not a finding.</li>
<li><b>What an edge means biologically.</b> It means the two genes rise and fall together across the nine samples, which is evidence that they are <b>co-regulated</b>, that is switched on and off by the same signal, perhaps the same transcription factor, or as part of the same pathway or protein complex. That is a genuinely useful hint, and it is the basis of the guilt-by-association reasoning used to guess what an uncharacterised gene does. But be careful about what it does <em>not</em> mean. It is not evidence that the two proteins touch each other, nor that one controls the other, nor that either is doing anything causal. It also does not take much to produce: with only 9 samples, and a shared response as strong as the 1-hour handling effect, plenty of gene pairs will correlate above 0.95 for reasons that have nothing to do with a shared function. Correlation networks generate hypotheses; they do not test them.</li>
<li><b>Relation to the hierarchical clusters. Yes, they largely agree, and they should.</b> Both are built from the same correlations between the same genes, so a set of genes that move together will show up as a branch in the tree and as a knot in the network. What differs is the shape of the answer. Hierarchical clustering forces every gene into exactly one group at whatever height you cut, whether or not it really belongs anywhere. The network drops genes with no strong partner altogether, so it is happy to leave a gene out, and it lets a gene sit between two knots and belong to both. So the network is better at showing you which genes are peripheral and which sit at a junction, while clustering is better when you want a tidy partition of every gene. Seeing the same grouping in both is reassuring, since it means the structure is in the data rather than in one method's assumptions.</li>
<li><b>Which genes to study next, and what experiment.</b> Central genes are a reasonable place to start, but say why rather than just naming the top of the list. A gene with high <em>degree</em> is correlated with many others, so it may sit at the heart of a co-regulated module; a gene with high <em>betweenness</em> sits on the paths between different knots, which makes it a candidate for connecting two processes. Both are hypotheses about position in a correlation graph, not measured facts about regulation, so the proposal has to be tested. A concrete suggestion: take two or three high-centrality genes from the knot that corresponds to the AVR-specific cluster, check what is already known about them in a database such as UniProt, obtain or generate knockout and overexpression lines, repeat the same infection experiment with AVR, VIR and MOCK, and measure both the disease outcome and the expression of the genes they were correlated with. If knocking the gene out flattens the response of its neighbours, you have evidence that it really is upstream of them rather than merely correlated with them. And since the aim is resilient crops rather than <em>Arabidopsis</em>, the last step is to find the equivalent gene in a crop species, which is what the translational biology practical does.</li>
</ul>""",
]


BADGE_OLD = '<div class="badge">Climate-Resilient Crops · Data Tutorial</div>'
BADGE_NEW = '<div class="badge">Climate-Resilient Crops · Answer Model</div>'

# Every student page closes its hero the same way: a "By the end of this practical"
# paragraph, the learning-goals list, and a single link back to the index. The answer
# page replaces all three with one intro paragraph and a two-link nav.
HERO_RE = re.compile(
    r' *<p style="text-align:left; max-width: \d+px; margin: 0 auto;">'
    r'By the end of this practical.*?</ul>\n'
    r' *<div class="nav">.*?</div>\n',
    re.S,
)

# Anchor for the answer-callout styling. Present in every student page, and the
# student pages deliberately do not carry a .callout.answer rule of their own.
ANSWER_CSS_ANCHOR = "  .callout.q strong { color: var(--accent-2); }\n"
ANSWER_CSS = """  .callout.answer { border-left-color: var(--accent); background: rgba(76,195,138,.09); color: var(--ink); margin-top: 12px; }
  .callout.answer strong { color: var(--accent); }
  .callout.answer ul { margin: 6px 0 0; }
"""


# name:  the practical's <h1>, used to locate it in the student page
# short: what the answer page calls itself, when the full <h1> is too long a title
# tail:  the sentence that follows "(green box)." in the answer page's intro
PAGES = {
    "marker_assisted_selection": {
        "name": "Marker Assisted Selection",
        "answers": MARKER_ASSISTED_SELECTION_ANSWERS,
        "tail": "All the interactive steps still work, so you can keep running the GWAS and "
                "the breeding programme while you read, and the self-test quiz at the end is unchanged.",
    },
    "genomic_selection": {
        "name": "Genomic Selection",
        "answers": GENOMIC_SELECTION_ANSWERS,
        "tail": "All the interactive steps still work, so you can keep retraining the model and "
                "running breeding cycles while you read, and the self-test quiz at the end is unchanged.",
    },
    "transcriptomics_clustering": {
        "name": "Transcriptomics &amp; Clustering",
        "answers": TRANSCRIPTOMICS_CLUSTERING_ANSWERS,
        "tail": "All the interactive steps still work, so you can keep re-clustering the data and "
                "exploring the network while you read, and the self-test quiz at the end is unchanged.",
    },
    "mechanistic_model": {
        "name": "Mechanistic Modelling of a Gene Regulatory Network",
        "short": "Mechanistic Modelling",
        "answers": MECHANISTIC_MODEL_ANSWERS,
        "tail": "All the interactive steps still work, so you can keep simulating droughts and "
                "knockouts while you read, and the self-test quiz at the end is unchanged.",
    },
    "deep_learning": {
        "name": "Deep Learning on Leaf Images",
        "answers": DEEP_LEARNING_ANSWERS,
        "tail": "All the interactive steps still work, so you can keep training models while you "
                "read, and the self-test quiz at the end is unchanged.",
    },
    "go_enrichment": {
        "name": "GO Enrichment Analysis",
        "answers": GO_ENRICHMENT_ANSWERS,
        "tail": "All the interactive steps still work, so you can keep clustering and re-running "
                "the analysis while you read, and the self-test quiz at the end is unchanged.",
    },
    "translational_biology": {
        "name": "Translational Biology",
        "answers": TRANSLATIONAL_BIOLOGY_ANSWERS,
        "tail": "All the interactive steps still work, so you can keep looking genes up and "
                "aligning proteins while you read, and the self-test quiz at the end is unchanged.",
    },
}


def _replace_once(html: str, old: str, new: str, where: str) -> str:
    """Substitute a fixed string, refusing to guess if it is not there exactly once."""
    if html.count(old) != 1:
        raise SystemExit(
            f"{where}: expected exactly one occurrence of {old[:60]!r}, found {html.count(old)}"
        )
    return html.replace(old, new)


def build_page(key: str) -> None:
    page = PAGES[key]
    src, dst = DOCS / f"{key}.html", DOCS / f"{key}_answers.html"
    answers, name = page["answers"], page["name"]
    short = page.get("short", name)
    html = src.read_text(encoding="utf-8")

    blocks = list(re.finditer(r'<div class="callout q">.*?</div>', html, re.S))
    if len(blocks) != len(answers):
        raise SystemExit(
            f"docs/{src.name} has {len(blocks)} question blocks but "
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

    # Title, heading and badge. The <title> is rewritten rather than suffixed, because
    # a couple of student pages carry an extra phrase there that the answer page drops.
    html, n = re.subn(r"<title>.*?</title>", f"<title>{short}: Answer Model</title>", html, count=1)
    if n != 1:
        raise SystemExit(f"docs/{src.name}: no <title> to replace")
    html = _replace_once(html, f"<h1>{name}</h1>", f"<h1>{short}: Answer Model</h1>", src.name)
    html = _replace_once(html, BADGE_OLD, BADGE_NEW, src.name)

    # Learning goals and the back-link become the answer-model intro and nav.
    hero = (
        f'  <p style="max-width: 660px; margin: 0 auto;">This is the answered version of the '
        f'{short} practical. Every ❓ question is followed by a worked model answer (green box). '
        f'{page["tail"]}</p>\n'
        f'  <div class="nav"><a href="{src.name}">&larr; Student version</a> &nbsp;·&nbsp; '
        f'<a href="index.html">All practicals</a></div>\n'
    )
    html, n = HERO_RE.subn(lambda _: hero, html, count=1)
    if n != 1:
        raise SystemExit(f"docs/{src.name}: could not find the hero block to replace")

    # Styling for the answer callouts, which only the generated page needs.
    html = _replace_once(html, ANSWER_CSS_ANCHOR, ANSWER_CSS_ANCHOR + ANSWER_CSS, src.name)

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
