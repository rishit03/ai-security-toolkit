# 🛡️ AI Security Toolkit

A red-team framework for testing the vulnerabilities of AI models through adversarial attacks, privacy leakage, and model exploitation techniques — built and maintained by [@rishit03](https://github.com/rishit03).

---

## 🚀 Features

✅ 5+ attack modules  
✅ Unified logging and visualization  
✅ Command-line interface (interactive menu)  
✅ Modular, reusable, and pip-installable  
✅ Built using TensorFlow, CleverHans, and Python's best practices

---

## 📦 Modules Included

| Module Name                | Description |
|----------------------------|-------------|
| 🔓 Adversarial Attack (FGSM)     | Confuses the model with small pixel changes |
| 💉 Label Flip Poisoning          | Modifies training labels to reduce model accuracy |
| 🧠 Membership Inference Attack  | Infers if a data point was used in training |
| 🪞 Model Inversion              | Reconstructs training images from the model |
| 🧬 Model Stealing               | Clones the target model using black-box queries |
| 🎯 Backdoor Trigger Attack      | Embeds a hidden trigger that forces misclassification |

---

## 💻 CLI Usage

```bash
# After pip install or cloning locally
python ai_toolkit/run.py
