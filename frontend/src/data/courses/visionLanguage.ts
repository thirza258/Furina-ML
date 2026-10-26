import type { Course } from "./types";

export const visionLanguageCourse: Course = {
  id: "vision-language", title: "Vision-Language Models (VLMs)", level: "Advanced", track: "Multimodal",
  description: "Learn how image and language representations connect. Cover contrastive training, visual instruction tuning, document understanding, video inputs and grounded evaluation.",
  prerequisites: ["Neural training and validation", "Transformers, embeddings and token losses", "Basic image tensors and Python"],
  prerequisiteCourses: ["transformers", "llms"], practiceMode: "python", dependencyTier: 3, nextCourseId: "voice",
  outcomes: ["Construct aligned image/text datasets", "Train and evaluate a contrastive representation model", "Explain visual connectors and instruction tuning", "Test grounding, OCR, hallucination and multimodal limits"],
  lab: { filename: "vision_language.py", title: "Align small images and text descriptions", description: "A CPU contrastive-learning lab with generated colored image patterns and short descriptions. It trains image and text encoders, uses a multi-positive objective for repeated descriptions, and evaluates held-out image-to-text retrieval. It is an alignment exercise, not a visual chat model.", exercises: ["Inspect image/text pairs and the set of valid matches.", "Compare retrieval before and after training on held-out images.", "Add a new visual attribute and test whether the descriptions and evaluation distinguish it."] },
  capstone: { title: "Build a grounded multimodal evaluation", brief: "Extend the alignment lab or adapt an appropriately licensed small VLM externally. Define what visual evidence each task requires and evaluate the full image/text system.", deliverables: ["An image/text manifest with source-level splits and valid-match labels", "Preprocessing, token-budget and trainable-module specifications", "Retrieval or VQA results with image-grounding controls", "A report of failures, resource use and unsupported input types"], rubric: ["Matching captions are not accidentally treated as negatives", "Evaluation sources are independent of training sources", "Answers or retrieved items depend on relevant image evidence", "Claims distinguish retrieval, OCR and generative visual reasoning"] },
  references: [
    { title: "CLIP — learning visual models from natural language supervision", url: "https://arxiv.org/abs/2103.00020" },
    { title: "LLaVA — visual instruction tuning", url: "https://arxiv.org/abs/2304.08485" },
    { title: "Hugging Face: image-text-to-text models", url: "https://huggingface.co/docs/transformers/tasks/image_text_to_text" },
  ],
  lessons: [
    {
      id: "multimodal-data", title: "Design paired visual and language data", minutes: 30,
      intro: "A vision-language model learns a relationship between visual inputs and language. That relationship must be present in the data, not merely assumed because an image and a caption share a row.",
      sections: [
        { title: "Choose the multimodal task", body: "Retrieval ranks images or captions by compatibility. Visual question answering produces an answer conditioned on an image and a question. OCR extracts visible text, while grounding connects language to image regions. These tasks need different labels, output structures and metrics; success at one does not automatically establish the others." },
        { title: "Audit the pairs", body: "Check whether captions describe visible content, whether question answers can be inferred from the image, and whether annotation text contains shortcuts. Track source image IDs, capture sessions, document families and usage permissions. Near-duplicate images or pages from the same document should stay together when their shared source would make evaluation easier." },
        { title: "Preserve evidence during preprocessing", body: "Resizing, cropping and compression can remove the small text or object that answers a question. Decide image resolution and augmentation based on the task. Keep a view of the processed image beside its label during data inspection. A model cannot learn evidence that the preprocessing has erased." },
      ],
      example: { title: "A broken visual question", body: "The question asks for a serial number in a corner, but the training crop removes that corner. Keeping the original answer encourages guessing from context. Repair the crop or remove the example rather than rewarding a prediction unsupported by its actual input." },
      practice: ["Inspect the generated image patterns and text descriptions in the lab.", "Design a manifest with image source, description, task and split.", "List which visual evidence would be lost by a center crop in an OCR dataset."],
      quiz: { question: "What should you do when preprocessing removes the visual evidence needed by a label?", options: ["Keep the label and assume the model will infer it", "Always increase the text length", "Repair the preprocessing or remove the invalid training example"], correctIndex: 2, explanation: "Supervision should be supported by the input the model actually sees. Missing evidence can train shortcuts or guesses." },
    },
    {
      id: "visual-encoders", title: "Represent images with patches and encoders", minutes: 30,
      intro: "Visual encoders turn pixels into features. The choice of feature resolution determines which details remain available to later language or retrieval components.",
      sections: [
        { title: "Track image and patch shapes", body: "An image batch commonly has [batch, channels, height, width]. A patch-based encoder divides an image into local regions and maps each region to a vector. With square images of side H and nonoverlapping patches of side P, there are (H/P)² patch positions when dimensions divide evenly, before any extra special tokens." },
        { title: "Distinguish pooled and spatial features", body: "A pooled embedding summarizes an image for retrieval or classification. A sequence of patch features retains more spatial detail for a language decoder or grounding task. Crops and tiling can improve coverage of small details but increase visual token counts. Record which representation is supplied to the next component." },
        { title: "Use the encoder's expected preprocessing", body: "Pretrained encoders expect particular pixel ranges, normalization, channel order and resize policies. Replacing them with arbitrary transforms can degrade representations even when tensor shapes match. Decide whether the encoder is frozen, partly unfrozen or trained from scratch and measure the resulting compute and generalization trade-off." },
      ],
      example: { title: "Patch-count arithmetic", body: "A 224×224 image divided into 16×16 patches gives 14×14 = 196 patch positions. A 448×448 image with the same patch size gives 784. In a naive dense-attention score matrix over these positions, that fourfold token increase means sixteen times as many entries." },
      practice: ["Calculate patch counts for two input resolutions.", "Explain when a pooled image embedding loses information needed for a spatial question.", "Inspect the lab's convolutional encoder and compare its output contract with a patch sequence."],
      quiz: { question: "What happens to patch count when both image dimensions double at a fixed patch size?", options: ["It becomes four times as large", "It stays unchanged", "It is cut in half"], correctIndex: 0, explanation: "There are twice as many patches across each dimension, giving four times as many total patch positions." },
    },
    {
      id: "contrastive-alignment", title: "Learn image/text alignment with contrastive loss", minutes: 40,
      intro: "Contrastive learning brings matching representations together relative to competing candidates. Correct positive and negative relationships are essential to the objective.",
      sections: [
        { title: "Build a similarity matrix", body: "Encode images and text into the same embedding width, normalize the embeddings and compute their pairwise dot products. A temperature scales the logits before a contrastive objective. With one unique pair per item, both image-to-text and text-to-image directions can train the system. Learned compatibility is meaningful relative to the candidate distribution used in training." },
        { title: "Handle multiple valid matches", body: "A batch may contain several images with the caption “red square”. Treating only the diagonal pair as positive incorrectly makes other correct matches negatives. Use a multi-positive objective, deduplicate appropriately or design batches with unambiguous matching labels. The supplied lab uses matching class descriptions as multiple positives rather than assuming every repeated caption is wrong." },
        { title: "Evaluate retrieval on held-out examples", body: "Recall@k checks whether at least one valid item appears among the top k candidates under the chosen relevance definition. Candidate pool size and valid-match labels matter. A retrieval score on training pairs can reflect memorization. Report the pool construction, both directions when relevant, and results for difficult or underrepresented concepts." },
      ],
      example: { title: "Two correct captions", body: "Two different photographs show red squares, and both captions read “red square”. A training objective that demands the first image select only the first caption punishes a semantically correct match. Define positives by valid relevance, not by row number alone." },
      practice: ["Run the vision-language lab and inspect its positive-match mask.", "Compare untrained and trained retrieval on its held-out images.", "Add a second phrasing for a caption and update the relevance definition."],
      quiz: { question: "Why can a diagonal-only contrastive target fail when captions repeat?", options: ["It disables the image encoder", "Other semantically valid matches may be treated as negatives", "It prevents any matrix multiplication"], correctIndex: 1, explanation: "Row identity is not always equivalent to semantic relevance. Repeated or equivalent descriptions require appropriate positive labels." },
    },
    {
      id: "visual-connectors", title: "Connect visual features to a language model", minutes: 35,
      intro: "A generative VLM must expose visual information to its language component. The connector defines how image features become usable context for text generation.",
      sections: [
        { title: "Match feature spaces and sequence layout", body: "A projector can map vision-encoder features into the language model's hidden width. Other architectures use learned queries or cross-attention to compress or access visual features. Decide where image tokens appear relative to text, how many are produced and which masks permit interaction. Matching vector widths alone does not ensure the representations are aligned." },
        { title: "Plan staged training", body: "One approach first trains a connector with fixed pretrained encoders, then adapts additional components on visual instruction data. The appropriate schedule depends on the architecture and starting checkpoints. Track which modules receive gradients at each stage and preserve the preprocessing expected by the vision encoder and language tokenizer." },
        { title: "Check that the image is actually used", body: "A language model can answer frequent questions from textual priors even when visual features are ignored. Compare outputs after replacing, removing or contradicting the image while holding the question fixed. Such controls reveal dependence on visual evidence, but must be interpreted carefully when the changed image makes a question unanswerable." },
      ],
      example: { title: "A projector contract", body: "Suppose the encoder produces 196 visual vectors of width 768 and the language model expects width 1,024. A projector maps [batch,196,768] to [batch,196,1024]. Sequence insertion, position handling and attention permissions must still be defined." },
      practice: ["Draw the image-encoder → connector → language-model path with shapes.", "Specify the trainable modules for an initial alignment stage and a later adaptation stage.", "Write an image-swap control for a simple visual question."],
      quiz: { question: "What does a visual projector primarily solve?", options: ["It guarantees that answers are grounded", "It creates missing image labels automatically", "It maps visual features into a representation compatible with the next component"], correctIndex: 2, explanation: "A projector connects feature spaces. Whether the model uses those features correctly must still be trained and evaluated." },
    },
    {
      id: "visual-instruction-tuning", title: "Train visual question answering and document tasks", minutes: 40,
      intro: "Visual instruction tuning combines an image, a request and a desired response. Its supervision must preserve both the image/text relationship and the language model's formatting rules.",
      sections: [
        { title: "Inspect the processed multimodal batch", body: "Use the model's documented processor to combine image preprocessing and tokenization. Verify that image placeholders correspond to the actual number and order of supplied images. Decode text and inspect processed images together. Incorrect placeholders, truncated image tokens or reversed image order can silently change the task." },
        { title: "Score answers while preserving context", body: "For response-only training, keep user text and visual features available while excluding non-target text and padding from the language loss. The treatment of visual tokens depends on the architecture. Confirm supervised positions after the processor and collator have expanded the sequence. Avoid examples whose answer disappeared after truncation." },
        { title: "Design data for the intended capability", body: "Captioning, OCR, counting, spatial reasoning and document extraction require different examples. Include unanswerable questions where the image genuinely lacks enough evidence, with a clear response policy. For structured extraction, evaluate schema validity separately from whether each extracted field matches visible content." },
      ],
      example: { title: "A receipt task", body: "The prompt asks for the receipt total as JSON. A valid JSON object containing the wrong total should fail content evaluation. Include small fonts, rotated receipts and missing totals in the suite if those inputs matter to the intended use." },
      practice: ["Design five VQA or document-extraction examples with explicit evidence.", "Identify prompt, image and answer positions in a planned processed batch.", "Define separate format and field-correctness checks for a structured answer."],
      quiz: { question: "What should a structured document answer be evaluated for?", options: ["Both schema validity and agreement with visible source content", "Only whether it is valid JSON", "Only whether it is longer than the prompt"], correctIndex: 0, explanation: "A syntactically valid answer can still invent or misread a field. Structure and visual correctness need separate checks." },
    },
    {
      id: "vlm-evaluation", title: "Measure grounding, OCR and hallucination", minutes: 35,
      intro: "Multimodal scores are meaningful only when they test the visual evidence required by the task. Fluent language can conceal failure to inspect the image.",
      sections: [
        { title: "Choose metrics for the output", body: "Retrieval uses relevance-based ranking metrics. OCR can use character or word error rates with documented normalization. Constrained VQA may use exact match or accepted answer sets. Grounding can evaluate spatial overlap against annotated regions. Open-ended caption quality needs more than string overlap when several descriptions are valid." },
        { title: "Use visual counterfactuals", body: "Compare questions on similar images that differ in the answer-relevant object, count or text. Add image-swap and text-only controls. Check whether a model changes its answer when the visual evidence changes. Keep the intervention realistic enough that a result can be interpreted as grounding rather than a response to an obviously corrupted input." },
        { title: "Inspect failures by slice", body: "Separate results by image quality, object size, language, document layout and other relevant conditions. Check unsupported claims about objects absent from the image and whether uncertainty is expressed when evidence is missing. Record annotation ambiguity and evaluator disagreements instead of forcing every open-ended answer into an unreliable binary score." },
      ],
      example: { title: "A text-only shortcut", body: "If nearly every training question “What color is the banana?” has answer “yellow”, a model may succeed without inspecting pixels. Include green and brown bananas, other objects and unanswerable cases to test whether its answers depend on visual evidence." },
      practice: ["Create a small suite with matched images whose correct answers differ.", "Define normalization rules for one OCR metric.", "Add a text-only baseline and explain what it reveals about the dataset."],
      quiz: { question: "What does a strong text-only baseline suggest in a supposedly visual task?", options: ["Visual grounding is guaranteed", "The dataset may contain language shortcuts", "The image encoder must be too large"], correctIndex: 1, explanation: "If the text alone predicts many answers, high task accuracy may not demonstrate use of visual evidence." },
    },
    {
      id: "video-and-resource-limits", title: "Extend to video and budget multimodal inputs", minutes: 30,
      intro: "Video and high-resolution documents increase the amount of visual context. Sampling policies and token budgets become part of what the model can observe.",
      sections: [
        { title: "Sample frames for the question", body: "Uniform frame sampling gives coverage but can miss a brief event. Dense local sampling may capture motion while omitting earlier context. Record frame timestamps and preserve order. Questions about temporal relationships require training and evaluation that expose those relationships; a collection of unordered images is not automatically a video-understanding system." },
        { title: "Count visual and text context together", body: "Multiple frames, crops or tiles add visual tokens that compete with text for memory and context. Estimate token counts before batching and profile representative long examples. A resolution that improves OCR may reduce throughput substantially. Choose limits using the intended task and failure cost rather than only fitting the largest possible input." },
        { title: "Document unsupported input conditions", body: "A still-image model may not support temporal reasoning, arbitrary numbers of frames or audio/video synchronization. A model trained on clean documents may fail on handwriting or severe blur. Reject, abstain or route unsuitable inputs according to the application's needs, and evaluate those behaviors explicitly." },
      ],
      example: { title: "A missed brief event", body: "Sampling one frame every five seconds can miss a one-second action entirely. No downstream reasoning can recover unseen evidence reliably. Change the sampling strategy or narrow the claim rather than blaming only the decoder." },
      practice: ["Calculate the visual-token budget for eight frames under your chosen representation.", "Design a sampling policy for a short-event detection question.", "List input limits and corresponding evaluation cases for a document or video application."],
      quiz: { question: "Why must frame sampling be included in a video model's evaluation protocol?", options: ["It only changes filenames", "It replaces the need for a visual encoder", "It determines which events are available as evidence"], correctIndex: 2, explanation: "A model can only reason over the observations supplied to it. Sampling can remove the event needed to answer a question." },
    },
    {
      id: "vlm-capstone", title: "Capstone: demonstrate grounded multimodal behavior", minutes: 55,
      intro: "Build a focused multimodal project and show what it learned using valid held-out examples and visual controls. Keep alignment experiments and generative claims distinct.",
      sections: [
        { title: "Choose retrieval or generation explicitly", body: "The supplied lab trains alignment between generated images and descriptions. Extend it with new attributes and relevance labels, or use an external VLM for a documented VQA task. State which task the artifact supports. A successful retrieval encoder does not become a conversational visual assistant merely by adding a text interface." },
        { title: "Evaluate a fully specified input path", body: "Save image normalization, crop policy, tokenizer or description vocabulary, architecture and selected checkpoint. Freeze candidate pools or VQA prompts before final testing. Use new source images and report relevance labels carefully when several answers are valid. Add at least one control that tests whether visual evidence affects the result." },
        { title: "Explain the limits", body: "Report task metrics, resource costs and examples of visual mistakes. Describe which attributes, resolutions, layouts or temporal conditions were not tested. Use the capstone rubric to review the practical output after passing the quiz. A useful project is small enough to reproduce and specific enough that its claims can be checked." },
      ],
      example: { title: "An honest alignment result", body: "“The encoders retrieve the matching color-and-pattern description for new generated images. We have not evaluated natural photographs, OCR or visual question answering.” Add measured retrieval scores and difficult examples from your own run." },
      practice: ["Complete the vision-language capstone checklist.", "Run the saved input pipeline on new held-out examples.", "Include an image-grounding control and explain one failure it exposes."],
      quiz: { question: "Which claim is supported by a successful image/text retrieval lab alone?", options: ["The learned embeddings rank the specified matching candidates under the tested conditions", "The model understands all natural photographs", "The model can answer arbitrary questions about videos"], correctIndex: 0, explanation: "Retrieval evidence supports the evaluated ranking task and distribution. Broader generative or temporal claims need additional training and evaluation." },
    },
  ],
};
