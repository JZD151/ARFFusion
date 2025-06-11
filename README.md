# ARFFusion

[![LICENSE](https://img.shields.io/badge/License-MIT-green)](https://github.com/lok-18/IGNet/blob/master/LICENSE)

### Adversarially Robust Fourier-Aware Multimodal Medical Image Fusion for LSCI
### ‼️*Environment* 

```shell
conda env create -f environment/environment.yml
conda activate LSCI
```

### 📑*Dataset setting*
> We give several test image pairs as examples in [[*Harvard Medical website*]](http://www.med.harvard.edu/AANLIB/home.html), respectively.
>
> Moreover, you can set your own test datasets of different modalities under ```./medicine_test/...```, like:   
> ```
> medicine_test
> ├── mri_ct
> │   ├── mri
> │   │   ├── 001.png
> │   │   ├── 002.png
> │   │   └── ...
> │   ├── oth
> │   │   ├── 001.png
> │   │   ├── 002.png
> │   │   └── ...
> │   └── Pseudo_label
> │       ├── 001.png
> │       ├── 002.png
> │       └── ...
> └── mri_pet
>     ├── mri
>     │   ├── 175.png
>     │   ├── 176.png
>     │   └── ...
>     ├── oth
>     │   ├── 175.png
>     │   ├── 176.png
>     │   └── ...
>     └── Pseudo_label
>         ├── 175.png
>         ├── 176.png
>         └── ...
> ```
>
> The configuration of the training dataset is similar to the aforementioned format.

### 🖥️*Test*
> The pre-trained model `model.pth` has given in [[*Google Drive*\]](https://drive.google.com/file/d/1X1QneHZeQNwpDKhDWbGICBV4h79EvP4e/view?usp=drive_link) and [[*Baidu Yun*\]](https://pan.baidu.com/s/1UDEu_Mwkl0G8TnOO8KSLlQ?pwd=968k).
>
> Please put ```model.pth``` into ```./checkpoint/``` and run ```test_robust.py``` to get fused results. You can check them in:
> ```
> test
> └── results
>     ├── mri_ct
>     │   ├── advmri
>     │   │   ├── 001.png
>     │   │   ├── 002.png
>     │   │   └── ...
>     │   ├── advoth
>     │   │   ├── 001.png
>     │   │   ├── 002.png
>     │   │   └── ...
>     │   └── fused
>     │       ├── 001.png
>     │       ├── 002.png
>     │       └── ...
>     └── mri_pet
>         ├── advmri
>         │   ├── 175.png
>         │   ├── 176.png
>         │   └── ...
>         ├── advoth
>         │   ├── 175.png
>         │   ├── 176.png
>         │   └── ...
>         └── fused
>             ├── 175.png
>             ├── 176.png
>             └── ...

### ⌛*Train*
> You can also utilize your own data to train a new robust fusion model with: 
> ```shell
> python train_robust.py
> ```

### 📬*Contact*
> If you have any questions, please create an issue or email to archerv2@mail.nwpu.edu.cn.
