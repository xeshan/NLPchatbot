import requests
import bs4
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
Tfidf_vect = TfidfVectorizer()

dialogs = {
    "hello_intent": "Hello! How are you doing? I am a chatbot that can help you answer questions in science. Try asking me one.",
    "whoami_intent":"Hi! I am a chatbot that can help you answer questions in science. Try asking me one.",
    "bye_intent": "Nice talking to you. Good day!",
}

def findbest(query, res_list):
    l = []
    for x in res_list:
        data = [query, x]
        vector_matrix = Tfidf_vect.fit_transform(data)
        cosine_similarity_matrix = cosine_similarity(vector_matrix)
        l.append(cosine_similarity_matrix[0][1])
    return l.index(max(l))

def perform_query(text):
    url = f"https://google.com/search?q={text}"
    headers = {"user-agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36 Edg/108.0.1462.46"}
    res = requests.get(url, headers=headers)
    parsed = bs4.BeautifulSoup(res.text, "html.parser")
    headings = parsed.find_all("h3")
    headingsT = [info.getText() for info in headings]
    wiki = 1 if "Description" in headingsT else 0
    featured = 1 if "About featured snippets" in parsed.body.text else 0

    if wiki:
        response = headings[headingsT.index("Description")].find_parent().find('span').text
    elif featured:
        response = parsed.body.find(text="About featured snippets").find_parent().find_parent().find_parent().find_parent().find_parent().find_parent().find_parent().find_parent().find('div').find_all('span')[0].text
    else:
        responses = [x.find_parent().find_parent().find_parent().find_parent().find_parent().find_all('span')[-1].text for x in headings]
        result = findbest(text, responses)
        response = responses[result][:-3]
        address = headings[result].find_parent().find_parent().find_parent().find_parent().find_parent().find('a').attrs["href"]
        
        best_res = bs4.BeautifulSoup(requests.get(address, headers=headers).text, "html.parser")
        if (f"{response[:50].strip()}" in best_res.body.text):
            response = best_res.body.find(text=re.compile(f"^.*{response[:50].strip()}.*$")).text

    return response
