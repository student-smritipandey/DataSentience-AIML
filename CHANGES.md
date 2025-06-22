# 🚀 Repository Reorganization

**Date: June 22, 2025**

---

⚠️ **IMPORTANT WARNING**:  
The repository structure has been completely reorganized for better clarity and future scalability.  
Please **update your imports and file paths accordingly** as some code may break due to path changes.

---

## 📂 Project Migration Summary

We've reorganized all projects into a cleaner and more logical structure. Below is the updated mapping:

---

### 🌾 Agricultural Solutions

| Previous Location                    | New Location                                          |
| ------------------------------------ | ----------------------------------------------------- |
| Agriculture/Crop yield predictor     | src/Agricultural Solutions/Crop Yield Prediction      |
| Agriculture/Crop recommender         | src/Agricultural Solutions/Crop Recommendation System |
| Agriculture/Plant disease protection | src/Agricultural Solutions/Plant Disease Detection    |

---

### 💰 Financial Analysis

#### 📈 Cryptocurrency

| Previous Location                    | New Location                                                    |
| ------------------------------------ | --------------------------------------------------------------- |
| Crypto Prediction/Bitcoin prediction | src/Financial Analysis/Cryptocurrency/Bitcoin Price Prediction  |
| Crypto Prediction/Wallet             | src/Financial Analysis/Cryptocurrency/Crypto Wallet Analysis    |
| Financial/Crypto sentimentalises     | src/Financial Analysis/Cryptocurrency/Crypto Sentiment Analysis |

#### 📊 Market Analysis

| Previous Location        | New Location                                                  |
| ------------------------ | ------------------------------------------------------------- |
| Financial/GDP prediction | src/Financial Analysis/Market Analysis/GDP Prediction         |
| Financial/Market rate    | src/Financial Analysis/Market Analysis/Market Rate Analysis   |
| Stock Price Analysis     | src/Financial Analysis/Market Analysis/Stock Price Prediction |

#### 🏢 Business Analytics

| Previous Location                  | New Location                                                        |
| ---------------------------------- | ------------------------------------------------------------------- |
| Financial/Startup profit predictor | src/Financial Analysis/Business Analytics/Startup Profit Prediction |
| Financial/Statement                | src/Financial Analysis/Business Analytics/Statement Analysis        |

---

### 🏥 Healthcare

#### 🧬 Disease Detection

| Previous Location                  | New Location                                              |
| ---------------------------------- | --------------------------------------------------------- |
| Health/Cancer                      | src/Healthcare/Disease Detection/Cancer Detection         |
| Health/Diabetes disease prediction | src/Healthcare/Disease Detection/Diabetes Prediction      |
| Health/Heart disease               | src/Healthcare/Disease Detection/Heart Disease Prediction |
| Health/Symptoms disease projector  | src/Healthcare/Disease Detection/Symptoms Analysis        |

#### 🖼️ Medical Imaging

| Previous Location         | New Location                                     |
| ------------------------- | ------------------------------------------------ |
| Federated Learning/Xray   | src/Healthcare/Medical Imaging/X-ray Analysis    |
| Health/Optical detections | src/Healthcare/Medical Imaging/Optical Detection |

#### 📟 Health Monitoring

| Previous Location               | New Location                                             |
| ------------------------------- | -------------------------------------------------------- |
| Health/Food classifier          | src/Healthcare/Health Monitoring/Food Classification     |
| Health/Mental health classifier | src/Healthcare/Health Monitoring/Mental Health Analysis  |
| Health/Students health products | src/Healthcare/Health Monitoring/Student Health Products |

---

### 🚗 Transportation & Safety

#### 👨‍✈️ Driver Safety

| Previous Location             | New Location                                                   |
| ----------------------------- | -------------------------------------------------------------- |
| Road Safety/Driver drowsiness | src/Transportation & Safety/Driver Safety/Drowsiness Detection |
| Road Safety/CCTV prediction   | src/Transportation & Safety/Driver Safety/CCTV Analysis        |

#### 🚦 Traffic Management

| Previous Location             | New Location                                                            |
| ----------------------------- | ----------------------------------------------------------------------- |
| Data Analysis/Traffic sign    | src/Transportation & Safety/Traffic Management/Traffic Sign Recognition |
| Road Safety/Ola bike requests | src/Transportation & Safety/Traffic Management/Ride Request Analysis    |

#### 🚙 Vehicle Analysis

| Previous Location        | New Location                                                       |
| ------------------------ | ------------------------------------------------------------------ |
| Financial/Car visibility | src/Transportation & Safety/Vehicle Analysis/Car Visibility System |

---

### 🌍 Environmental Monitoring

| Previous Location                | New Location                                    |
| -------------------------------- | ----------------------------------------------- |
| Data Analysis/Weather prediction | src/Environmental Monitoring/Weather Prediction |
| Data Analysis/Fire detection     | src/Environmental Monitoring/Fire Detection     |

---

### 🧠 Machine Learning Techniques

| Previous Location            | New Location                                       |
| ---------------------------- | -------------------------------------------------- |
| Federated Learning/Federated | src/Machine Learning Techniques/Federated Learning |
| Bidirectional STM            | src/Machine Learning Techniques/Bidirectional STM  |
| Health/OCR                   | src/Machine Learning Techniques/OCR Systems        |

---

## 🆕 Adding New Projects

### Available Categories

- **Agricultural Solutions**: Crop yield, crop recommendations, plant diseases
- **Financial Analysis**: Cryptocurrency, market analysis, business analytics
- **Healthcare**: Disease detection, medical imaging, health monitoring
- **Transportation & Safety**: Driver safety, traffic management, vehicle analysis
- **Environmental Monitoring**: Weather prediction, fire detection
- **Machine Learning Techniques**: Specialized ML methodologies
- **Miscellaneous**: Everything else

---

### How to Add a New Project

1. Choose the most suitable category
2. Create your project folder inside `src/[Category]/[Your Project Name]`
3. Submit a pull request (PR) with your additions
4. If your project doesn't fit, place it temporarily under:  
   `src/Miscellaneous/`
5. Open an issue if you'd like a new category added

---

### 🤖 AI Assistant Prompt for Categorization

Use this prompt to classify new projects:

```
I'm working on a new AI/ML project titled "[YOUR PROJECT TITLE]" for the repository.
The repository has the following categories:
- Agricultural Solutions
- Financial Analysis
- Healthcare
- Transportation & Safety
- Environmental Monitoring
- Machine Learning Techniques
- Miscellaneous

Based on the title, please suggest:
1. The most appropriate category
2. A suitable subcategory (if applicable)
3. A standardized name for the project folder
```

---

### 🗂️ Requesting a New Category

If no category fits:

- Use `src/Miscellaneous/`
- Open an issue with:
  - Project name & description
  - Reason existing categories don't fit
  - Suggested new category & potential subcategories
  - Any future projects that would belong

---

### 📊 Project Count

- **Original Structure**: 31 loosely organized projects across 9 folders
- **New Structure**: 31 projects in 6 structured categories with subcategories

---

### 🧱 Note on Shared Components

The `units` folder is now located at:  
**src/units/** for better modular structure and reuse.

---
