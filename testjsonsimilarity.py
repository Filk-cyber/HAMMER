import torch

from retrievers.e5_mistral import get_e5_mistral_embeddings_for_query, get_e5_mistral_embeddings_for_document
query=["Are the genuses Michelia and Stenomesson in the same family?"]
context=["Michelia ingrata is a species of plant in the Magnoliaceae family. It is endemic to China.",
         "Michelia wilsonii is a species of plant in the Magnoliaceae family. It is endemic to China. It is threatened by habitat loss.",
         "Michelia odora is a species of plant in the Magnoliaceae family. It is found in China and Vietnam. It is threatened by habitat loss.",
         "Stenomesseae was a tribe (in the family Amaryllidaceae, subfamily Amaryllidoideae), where it forms part of the Andean clade, one of two American clades. The tribe was originally described by Traub in his monograph on the Amaryllidaceae in 1963, as Stenomessae based on the type genus \"Stenomesson\". In 1995 it was recognised that Eustephieae was a distinct group separate from the other Stenomesseae. Subsequently, the M\u00fcller-Doblies' (1996) divided tribe Eustephieae into two subtribes, Stenomessinae and Eustephiinae.",
         "Stenomesson is a genus of bulbous plants in the family Amaryllidaceae. All the species are native to western South America (Colombia, Ecuador, Peru and northern Chile).",
         "Clinantheae is a tribe (in the family Amaryllidaceae, subfamily Amaryllidoideae), where it forms part of the Andean clade, one of two American clades. The tribe was described in 2000 by Alan Meerow \"et al.\" as a result of a molecular phylogenetic study of the American Amaryllidoideae. This demonstrated that the Stenomesseae tribe, including the type genus \"Stenomesson\" was polyphyletic. Part of the tribe segregated with the Eucharideae and were submerged into it, while the other part formed a unique subclade. Since the type species of \"Stenomesson\" was not part of the second sublclade it was necessary to form a new name for the remaining species together with the other genera that remained. This was \"Clinanthus\", the oldest name for these species, and consequently the tribe Clinantheae.",
         "Michelia punduana is a species of plant in the Magnoliaceae family. It is endemic to the Meghalaya subtropical forests in India. It is threatened by habitat loss.",
         "Michelia hypolampra is a species of plant in the Magnoliaceae family. It is endemic to China. It is threatened by habitat loss.",
         "Michelia is a genus of flowering plants belonging to the Magnolia family (Magnoliaceae). The genus includes about 50 species of evergreen trees and shrubs, native to tropical and subtropical south and southeast Asia (Indomalaya), including southern China.",
         "Michelia coriacea is a species of flowering plant in the family Magnoliaceae. It is native to China and Vietnam. There are no more than about 500 individuals remaining of this endangered species."]
qe=get_e5_mistral_embeddings_for_document(query)
ce=get_e5_mistral_embeddings_for_document(context)
qe = torch.nn.functional.normalize(qe, p=2, dim=-1)
ce = torch.nn.functional.normalize(ce, p=2, dim=-1)
print(ce.shape)
print("====================")
print(ce[:-3].shape)
si=torch.matmul(qe,ce.T).squeeze(0)
print(si)
print(si[7+0])
print(si[7+1])
print(si[7+2])
truthful_scores=[7,8,7,6,5,4,3,10,1,0]
truthful_scores_tensor = torch.tensor(truthful_scores, dtype=torch.bfloat16)
final_scores = si * truthful_scores_tensor
print(final_scores)
for scores in final_scores:
    if scores <= 3:
        credibility = "Low credibility of text"
    elif scores > 3 and scores < 7:
        credibility = "Medium credibility of text"
    elif scores >= 7:
        credibility = "High credibility of text"
    print(credibility)
topk_indices=torch.topk(si, k=5, dim=0)[1].tolist()
print(topk_indices)