# Rebuttal
We have consolidated all content onto a single page for the convenience of reviewers during the rebuttal period. The first half of this page contains all the materials requested by the reviewers that could not be accommodated within the one-page rebuttal file. The latter half presents the original GitHub assets associated with this project.

## More 

### AiATrack
>If you find the images on this anonymous GitHub page too small, you can right-click on the image, select "Open image in new tab," and then download the file. All images are in SVG format, which can be scaled up without loss of quality.

![UIR Figure 1](sup/pics/UIR/fig1.svg)

![UIR Figure 2](sup/pics/UIR/fig2.svg)

### ARTrack
>If you find the images on this anonymous GitHub page too small, you can right-click on the image, select "Open image in new tab," and then download the file. All images are in SVG format, which can be scaled up without loss of quality.

![UIR Figure 3](sup/pics/UIR/fig3.svg)

![UIR Figure 4](sup/pics/UIR/fig4.svg)

![UIR Figure 5](sup/pics/UIR/fig5.svg)

![UIR Figure 6](sup/pics/UIR/fig6.svg)

### Mixformer
>If you find the images on this anonymous GitHub page too small, you can right-click on the image, select "Open image in new tab," and then download the file. All images are in SVG format, which can be scaled up without loss of quality.

![UIR Figure 7](sup/pics/UIR/fig7.svg)

# Divider Line
> :point_up_2: rebuttal ------------------------------------------ divider line ------------------------------------------ :point_down: Github Assets

# Strong Neck

The official implementation for the paper Tracker is Filter: A Bottleneck Structure for Visual Object Tracking

## Abstract
> Can traditional signal processing methods shine again in the era of deep learning? Recent advancements in visual object tracking with transformer-based models have undeniably boosted performance and robustness. Yet, many approaches still add significant complexity through new modules, often leading to higher computational demands without a guaranteed performance boost. In our groundbreaking work, we've uncovered a fascinating characteristic of well-trained vision transformer-based models: distinct unit impulse responses across different layers. This implies that each layer might have a unique passband for processing input features. To test this hypothesis, we've developed a novel Bottleneck structure, 'Strong Neck', which comprises just 5 processing steps and contains no learnable parameters. By seamlessly integrating this module into well-trained trackers, our method has not only achieved improved performance across 7 datasets but also set new state-of-the-art (SOTA) results on 6 of them. This remarkable achievement highlights the effectiveness of 'Strong Neck' in enhancing tracking performance with minimal computational overhead. Our code is available at Anonymized Repository.
