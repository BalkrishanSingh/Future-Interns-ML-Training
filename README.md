# Future Interns machine learning training and semantic resume screener

Student: Balkrishan Singh (URN: 2302492)
Program: TR-103 Four Week Industrial Training (Machine Learning)
Institution: Department of Computer Science and Engineering, Guru Nanak Dev Engineering College, Ludhiana
Training Organization: Future Interns Development Center
Period: 26 June 2026 to 24 July 2026

## Repository structure

```text
├── Project_01_Sales_Forecasting/             # Week 1: Time series regression with Random Forest
│   ├── Sample - Superstore.csv
│   ├── sales_forecast.py
│   └── README.md
├── Project_02_Ticket_Classification/         # Week 2: Deep learning NLP support ticket classifier (LSTM)
│   ├── customer_support_tickets.csv
│   ├── ticket_classification_colab.py
│   └── README.md
├── Project_03_Semantic_Resume_Screener/      # Weeks 3 and 4: Capstone Semantic Screening & Talent Search Portal
│   ├── ml_pipeline/                          # Fine-tuned all-MiniLM-L6-v2, MNRL loss, ONNX export, CLI screener
│   ├── web_portal/                           # Next.js 15 App Router, client-side PDF extraction, PII scrubbing, dynamic sliders
│   ├── demo_resumes/                         # Multi-format candidate resumes for local CLI evaluation
│   ├── sample_resumes/                       # Test PDFs for web portal verification
│   ├── model_quantized.onnx                  # Exported INT8 quantized ONNX model for CPU inference
│   └── Resume.csv                            # 2.4k resume dataset
├── requirements.txt
└── README.md
```

## Four-week training timeline and completed work

- Week 1 (26 Jun to 01 Jul): Exploratory data analysis and retail sales forecasting using Random Forest regression with chronological validation splitting.
- Week 2 (02 Jul to 06 Jul): Support ticket classification using padded token sequences and a dual-branch LSTM recurrent network.
- Week 3 (07 Jul to 17 Jul): Major project engineering: dataset category mapping, domain fine-tuning of all-MiniLM-L6-v2 with Multiple Negatives Ranking Loss (MNRL), INT8 ONNX quantization, and a standalone CLI screener.
- Week 4 (20 Jul to 24 Jul): Interactive Next.js talent search portal with client-side PDF extraction, regex PII scrubbing, composite scoring with dynamic sliders, and unit test verification.
