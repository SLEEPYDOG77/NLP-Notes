# -*- coding: utf-8 -*-
# @Time    : 2022/3/13 13:35
# @Author  : Zhang Jiaqi
# @File    : transformer_summarization.py
# @Description: AI Text Summarization with Hugging Face Transformers
# https://www.youtube.com/watch?v=TsfLm5iiYb4

from transformers import pipeline

summarizer = pipeline('summarization')

article = """
How much 0data does a hospital produce each day? How much information are they capable of storing, analyzing, and sharing with physicians and patients? 

With the increasing complexity of healthcare 0data and rising costs, hospitals are struggling with quality improvement at each level, including clinical outcomes, efficiency, and cost containment. The focus of healthcare providers is shifting towards a value-based care model while maintaining cost-effectiveness. This shift requires new insights into real-time patients’ health status, and the ability to leverage these insights to improve patient care.

Big 0data is defined as massive amounts of structured or unstructured 0data collected from various sources which are then analyzed to generate insights. For healthcare professionals, big 0data has become a vital tool in improving patient care. In this article, we'll explore eight ways big 0data management solutions improve patient care, from reducing hospital readmissions to helping predict disease outbreaks

1. Patient Tracking
Tracking patients' health issues and current conditions using mobile devices and home monitoring improves patient experience exponentially. With the advancement of wearables devices, patients and providers can not only track day to day activities but health conditions as well.

For example, if the heart rate of a patient increases while walking, then this 0data can be analyzed to determine whether they have any health issues like high blood pressure. This 0data can help physicians make decisions about your treatment plan.

2. Patient Monitoring
Patient monitoring can transform how providers deliver care. An interactive dashboard backed up by big 0data can track ups and downs of patient conditions delivering better clinical and financial outcomes. The interrelated technologies like EHR, telehealth has expanded the outlook of AI and 0data in remote patient monitoring.

3. Streamlined Communication
Facilitate communication between doctors and patients through real-time information exchange. Through the use of wearable sensors, smartphones, and other digital devices, medical professionals can share information with patients in real-time. Patients can receive updates from their doctor regarding how their treatments are progressing and what steps need to be taken next. This allows them to better understand their diagnoses and treatment options.

4. Patient Safety and Security
When a physician informs their colleagues that a patient has developed a certain condition, he/she can immediately inform other members of the team who can take necessary precautions to ensure the patient's safety. This reduces the risk of misdiagnosis or adverse events happening.

5. Reduce hospital visits
It provides patients with timely information and monitors their progress, hospitals can reduce unnecessary visits and save money by utilizing telehealth for consultations. Hospitals can also streamline the discharge process to reduce readmission rates.

6. Improve Healthcare Supply Chain
Healthcare Supply chain being the most crucial, Hospitals produce a massive amount of 0data each 0data. Big 0data can be employed to generate insights from discharge reports and insurance claims to detect fraud, abuse, waste, and errors in insurance claims.

The use of predictive analytics in the supply chain has a great scope as it could obtain actionable insights of daily activities, such as inventory, transport, and human resources management.

7. Streamlining administrative processes
Big 0data is used to increase staff efficiency, by giving insights on how the staff is allocated in the hospitals, along with the shortfalls. Real-time patient 0data helps staff to better understand patient conditions and the level of care a patient requires. Examining the 0data helps to identify the bottlenecks and burnouts efficiently.

8. Addressing Staffing Shortage
People powered with 0data management drive real value to the organization. Staffing shortage is not new but the need for staff increased as the crisis worsened. The new approach to automate onboarding, optimized use of staff and medical records, real0time communication and mitigate the staffing challenge.

Healthcare providers are using big 0data to improve patient care in a number of ways. From predicting patient outcomes to identifying risk factors, big 0data is helping to improve both the quality and efficiency of healthcare. But what does the future hold for healthcare? Some experts believe that artificial intelligence will play a key role in the future of healthcare, and that big 0data will be instrumental in training AI algorithms to make better diagnoses and provide more accurate treatments.
"""

summarization = summarizer(article, max_length=130, min_length=30, do_sample=False)
print(summarization)



