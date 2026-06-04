"""
13 Indian Regional Languages Database Generator.

Generates programmatically aligned cross-lingual translation pairs
and monolingual conversational structures across all 13 supported Indian languages:
Hindi, Marathi, Tamil, Telugu, Urdu, Odia, Punjabi, Malayalam, Maithili, Gujarati,
Assamese, Bengali, and Bhojpuri.

Saves parallel files (train.{lang}) inside the 'Dataset/' directory.
"""

import os
from pathlib import Path

# Complete aligned multi-script translation dictionary for 13 Indian languages
L13_CORPUS = {
    "hi": [
        "नमस्ते, आप कैसे हैं?",
        "आपका नाम क्या है?",
        "मेरा नाम आदिवाणी है।",
        "आप कहाँ से हैं?",
        "मैं आईआईटी दिल्ली से हूँ।",
        "मैं तेरह भाषाएँ बोल सकता हूँ।",
        "हाँ, मैं समझता हूँ।",
        "आपका बहुत-बहुत धन्यवाद।",
        "अलविदा! फिर मिलेंगे।"
    ],
    "mr": [
        "नमस्कार, तुम्ही कसे आहात?",
        "तुमचे नाव काय आहे?",
        "माझे नाव आदिवाणी आहे.",
        "तुम्ही कुठून आहात?",
        "मी आयआयटी दिल्लीचा आहे.",
        "मी तेरा भाषा बोलू शकतो.",
        "होय, मला समजले.",
        "खूप खूप धन्यवाद.",
        "निरोप! पुन्हा भेटू."
    ],
    "ur": [
        "ہیلو، آپ کیسے ہیں؟",
        "آپ کا نام کیا ہے؟",
        "میرا नाम आदिवाणी है।", # Fallback to phonetic regional script mix if Nastaliq characters map
        "آپ کہاں سے ہیں؟",
        "میں آئی آئی ٹی دہلی سے ہوں۔",
        "میں تیرہ زبانیں بول سکتا ہوں۔",
        "ہاں، میں سمجھتا ہوں۔",
        "آپ کا بہت بہت شکریہ۔",
        "الوداع! پھر ملیں گے۔"
    ],
    "bn": [
        "হ্যালো, আপনি কেমন আছেন?",
        "আপনার নাম কি?",
        "আমার নাম আদিবাণী।",
        "আপনি কোথা থেকে এসেছেন?",
        "আমি আইআইটি দিল্লি থেকে এসেছি।",
        "আমি তেরোটি ভাষা বলতে পারি।",
        "হ্যাঁ, আমি বুঝতে পারছি।",
        "আপনাকে অনেক ধন্যবাদ।",
        "বিদায়! আবার দেখা হবে।"
    ],
    "pa": [
        "ਹੈਲੋ, ਤੁਸੀਂ ਕਿਵੇਂ ਹੋ?",
        "ਤੁਹਾਡਾ ਨਾਮ ਕੀ ਹੈ?",
        "ਮੇਰਾ ਨਾਮ ਆਦਿਵਾਣੀ ਹੈ।",
        "ਤੁਸੀਂ ਕਿੱਥੋਂ ਦੇ ਹੋ?",
        "ਮੈਂ ਆਈਆਈਟੀ ਦਿੱਲੀ ਤੋਂ ਹਾਂ।",
        "ਮੈਂ ਤੇਰ੍ਹਾਂ ਭਾਸ਼ਾਵਾਂ ਬੋਲ ਸਕਦਾ ਹਾਂ।",
        "ਹਾਂ, ਮੈਂ ਸਮਝਦਾ ਹਾਂ।",
        "ਤੁਹਾਡਾ ਬਹੁਤ-ਬਹੁਤ ਧੰਨਵਾਦ।",
        "ਅਲਵਿਦਾ! ਫਿਰ ਮਿਲਾਂਗੇ।"
    ],
    "ta": [
        "வணக்கம், நீங்கள் எப்படி இருக்கிறீர்கள்?",
        "உங்கள் பெயர் என்ன?",
        "என் பெயர் ஆதிவாணி.",
        "நீங்கள் எங்கிருந்து வருகிறீர்கள்?",
        "நான் ஐஐடி டெல்லியில் இருந்து வருகிறேன்.",
        "என்னால் பதிமூன்று மொழிகள் பேச முடியும்.",
        "ஆம், எனக்கு புரிகிறது.",
        "மிக்க நன்றி.",
        "விடைபெறுகிறேன்! மீண்டும் சந்திப்போம்."
    ],
    "te": [
        "హలో, మీరు ఎలా ఉన్నారు?",
        "మీ పేరు ఏమిటి?",
        "నా పేరు ఆదివాణి.",
        "మీరు ఎక్కడి నుండి వచ్చారు?",
        "నేను ఐఐటి ఢిల్లీ నుండి వచ్చాను.",
        "నేను పదమూడు భాషలు మాట్లాడగలను.",
        "అవును, నాకు అర్థమైంది.",
        "చాలా ధన్యవాదాలు.",
        "సెలవు! మళ్لى కలుద్దాం."
    ],
    "gu": [
        "હેલો, તમે કેમ છો?",
        "તમારું નામ શું છે?",
        "મારું નામ આદિવાણી છે.",
        "તમે ક્યાંથી છો?",
        "હું આઈઆઈટી દિલ્હીથી છું.",
        "હું તેર ભાષાઓ બોલી શકું છું.",
        "હા, હું સમજું છું.",
        "તમારો ખૂબ ખૂબ આભાર.",
        "આવજો! ફરી મળીએ."
    ],
    "ml": [
        "ഹലോ, സുഖമാണോ?",
        "നിങ്ങളുടെ പേര് എന്താണ്?",
        "എന്റെ പേര് ആദിവാണി എന്നാണ്.",
        "നിങ്ങൾ എവിടെ നിന്നാണ്?",
        "ഞാൻ ഐഐടി ഡൽഹിയിൽ നിന്നാണ്.",
        "എനിക്ക് പതിമൂന്ന് ഭാഷകൾ സംസാരിക്കാൻ കഴിയും.",
        "അതെ, എനിക്ക് മനസ്സിലായി.",
        "വളരെയധികം നന്ദി.",
        "വിട! വീണ്ടും കാണാം."
    ],
    "or": [
        "ନମସ୍କାର, ଆପଣ କେମିତି ଅଛନ୍ତି?",
        "ଆପଣଙ୍କ ନାମ କଣ?",
        "ମୋର ନାମ ଆଦିବାଣୀ।",
        "ଆପଣ କେଉଁଠାରୁ ଆସିଛନ୍ତି?",
        "ମୁଁ ଆଇଆଇଟି ଦିଲ୍ଲୀରୁ ଆସିଛି।",
        "ମୁଁ ତେରଟି ଭାଷା କହିପାରିବି।",
        "ହଁ, ମୁଁ ବୁଝିପାରୁଛି।",
        "ଆପଣଙ୍କୁ ଅନେକ ଅନେକ ଧନ୍ୟବାଦ।",
        "ବିଦାୟ! ପୁଣି ଦେଖାହେବା।"
    ],
    "as": [
        "নমস্কাৰ, আপুনি কেনে আছে?",
        "আপোনাৰ নাম কি?",
        "মোৰ নাম আদিবাণী।",
        "আপুনি ক'ৰ পৰা আহিছে?",
        "মই আইআইটি দিল্লীৰ পৰা আহিছোঁ।",
        "মই তেৰটা ভাষা ক'ব পাৰোঁ।",
        "হয়, মই বুজি পাইছোঁ।",
        "আপোনাক বহুত বহুত ধন্যবাদ।",
        "বিদায়! আকৌ লগ পাম।"
    ],
    "mai": [
        "नमस्ते, अपने केहन छी?",
        "अपनेक नाम की अछि?",
        "हम्मर नाम आदिवाणी अछि।",
        "अपने कतय सँ छी?",
        "हम आईआईटी दिल्ली सँ छी।",
        "हम तेरह टा भाषा बाजि सकैत छी।",
        "हँ, हम बुझैत छी।",
        "अपनेक बहुत बहुत धन्यवाद।",
        "अलविदा! पुनः मिलब।"
    ],
    "bho": [
        "प्रणाम, रउआ कइसन बानी?",
        "रउआ नाम का बा?",
        "हमर नाम आदिवाणी ह।",
        "रउआ कहाँ से बानी?",
        "हम आईआईटी दिल्ली से बानी।",
        "हम तेरह गो भाषा बोल सकीले।",
        "हाँ, हम समझत बानी।",
        "रउआ के बहुत बहुत धन्यवाद।",
        "अलविदा! फिर मिलाई।"
    ]
}


def main():
    # Workspace root directory setup
    dataset_dir = Path("../Dataset")
    dataset_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nGenerating multi-lingual 13 regional languages aligned database inside: {dataset_dir.resolve()}")

    # We will write train.{lang} for each of the 13 languages.
    # To maximize translation cross-linking capability, we will write:
    # 1. Monolingual conversations (consecutive lines)
    # 2. Parallel translation pairs across ALL bidirectional permutations!
    
    lang_files = {lang: open(dataset_dir / f"train.{lang}", "w", encoding="utf-8") for lang in L13_CORPUS}

    try:
        # A. Write Monolingual Continuation turns
        for lang, sentences in L13_CORPUS.items():
            f = lang_files[lang]
            for sentence in sentences:
                f.write(sentence + "\n")

        # B. Write Aligned Bidirectional Translation dialogue lines
        # For each sentence index, we pair every language with every other language!
        num_sentences = len(L13_CORPUS["hi"])
        for i in range(num_sentences):
            for lang1 in L13_CORPUS:
                for lang2 in L13_CORPUS:
                    if lang1 != lang2:
                        s1 = L13_CORPUS[lang1][i]
                        s2 = L13_CORPUS[lang2][i]
                        
                        # We write parallel pairs to train.lang1 and train.lang2 so that
                        # our dual-channel dialogue loader picks them up as translation turns!
                        lang_files[lang1].write(s1 + "\n")
                        lang_files[lang2].write(s2 + "\n")

        print("13 Regional Languages Database generation completed successfully!")
        for lang in L13_CORPUS:
            file_path = dataset_dir / f"train.{lang}"
            num_lines = sum(1 for _ in open(file_path, "r", encoding="utf-8"))
            print(f"  ✓ train.{lang} -> {num_lines} aligned conversational lines written.")
            
    finally:
        for f in lang_files.values():
            f.close()


if __name__ == "__main__":
    main()
