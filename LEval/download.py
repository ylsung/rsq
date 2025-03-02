from datasets import load_dataset, disable_caching

datasets = ["coursera", "gsm100", "quality", "topic_retrieval_longchat", "tpo", "codeU", "sci_fi" ,"financial_qa", "gov_report_summ", "legal_contract_qa", "meeting_summ", "multidoc_qa", "narrative_qa", "natural_question", "news_summ", "paper_assistant", "patent_summ", "review_summ", "scientific_qa", "tv_show_summ"]
# The corresponding NAMEs in the paper
# "coursera", "GSM(16-shot)", "QuALITY", "TopicRet", "TOFEL", "codeU", "SFiction", "LongFQA", "GovReport", "CUAD", "QMSum", "MultiDoc2Dial", "NarrativeQA", "NQ", "Multi-news", "Openreview", "BigPatent", "SPACE", "Qasper", "SummScreen"]

for testset in datasets:
    # disable_caching()  uncomment this if you cannot download codeU and sci_fi 
    data = load_dataset('L4NLP/LEval', testset, split='test')
    # evaluate your model
