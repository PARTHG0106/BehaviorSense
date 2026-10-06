/* Caregiver language preferences. The research document itself remains English. */
export const LANGUAGES = Object.freeze({ en: "English", hi: "हिन्दी", mr: "मराठी" });
const STORAGE_KEY = "behaviorsense.report-language";
let language = "en";
const listeners = new Set();
let boundSelect = null;

// Each row is English, Hindi, Marathi. Numeric placeholders are never reformatted.
const rows = {
  languageLabel: ["Report language", "रिपोर्ट की भाषा", "अहवालाची भाषा"],
  auditDetails: ["Evidence and original English report", "प्रमाण और मूल अंग्रेज़ी रिपोर्ट", "पुरावे आणि मूळ इंग्रजी अहवाल"],
  liveGenerate: ["Generate live", "नई रिपोर्ट बनाएँ", "नवा अहवाल तयार करा"],
  generateLive: ["Generate live", "नई रिपोर्ट बनाएँ", "नवा अहवाल तयार करा"],
  chooseFile: ["Choose a file", "फ़ाइल चुनें", "फाइल निवडा"],
  dropClip: ["Drop a clip here, or", "वीडियो यहाँ छोड़ें, या", "व्हिडिओ येथे टाका, किंवा"],
  enrolSubject: ["Enrol the subject as", "व्यक्ति को इस नाम से पंजीकृत करें", "व्यक्तीची या नावाने नोंदणी करा"],
  enrolPlaceholder: ["Leave blank to analyse without enrollment", "बिना पंजीकरण विश्लेषण के लिए खाली छोड़ें", "नोंदणीशिवाय विश्लेषणासाठी रिकामे ठेवा"],
  connect: ["Connect", "जोड़ें", "जोडा"],
  forget: ["Forget", "कनेक्शन हटाएँ", "जोडणी विसरा"],
  backendUrlPlaceholder: ["http://127.0.0.1:8899", "http://127.0.0.1:8899", "http://127.0.0.1:8899"],
  tokenPlaceholder: ["For a protected direct notebook connection", "सुरक्षित नोटबुक से सीधे जुड़ने के लिए", "संरक्षित नोटबुकशी थेट जोडण्यासाठी"],
  uploadConnect: ["Connect a backend to analyse a clip.", "वीडियो का विश्लेषण करने के लिए सेवा से जुड़ें।", "व्हिडिओचे विश्लेषण करण्यासाठी सेवेशी जोडा."],
  uploadUnavailable: ["Video analysis is unavailable on this backend.", "इस सेवा पर वीडियो विश्लेषण उपलब्ध नहीं है।", "या सेवेत व्हिडिओ विश्लेषण उपलब्ध नाही."],
  uploadBusy: ["This clip is being analysed. Please wait before choosing another.", "इस वीडियो का विश्लेषण हो रहा है। दूसरा चुनने से पहले प्रतीक्षा करें।", "या व्हिडिओचे विश्लेषण सुरू आहे. दुसरा निवडण्यापूर्वी थांबा."],
  uploadTooLarge: ["Choose a video smaller than 60 MB.", "60 MB से छोटा वीडियो चुनें।", "60 MB पेक्षा लहान व्हिडिओ निवडा."],
  uploadNotVideo: ["Choose a video file.", "वीडियो फ़ाइल चुनें।", "व्हिडिओ फाइल निवडा."],
  uploadRunning: ["Analysing {name}. Longer clips can take several minutes.", "{name} का विश्लेषण हो रहा है। लंबे वीडियो में कई मिनट लग सकते हैं।", "{name} चे विश्लेषण सुरू आहे. मोठ्या व्हिडिओंसाठी काही मिनिटे लागू शकतात."],
  uploadStage1: ["Poses extracted. Recognising activities…", "शरीर की मुद्राएँ मिल गई हैं। गतिविधियों की पहचान हो रही है…", "शरीराच्या मुद्रा मिळाल्या आहेत. हालचाली ओळखल्या जात आहेत…"],
  uploadStage2: ["Activities recognised. Measuring behaviour…", "गतिविधियों की पहचान हो गई है। व्यवहार का आकलन हो रहा है…", "हालचाली ओळखल्या आहेत. वर्तनाचे मोजमाप सुरू आहे…"],
  uploadStage3: ["Behaviour measured. Preparing and checking the report…", "व्यवहार का आकलन हो गया है। रिपोर्ट बनाई और जाँची जा रही है…", "वर्तनाचे मोजमाप झाले. अहवाल तयार करून तपासला जात आहे…"],
  uploadNoClaims: ["Analysis complete. No verifiable findings were produced.", "विश्लेषण पूरा हुआ। जाँच योग्य निष्कर्ष नहीं मिले।", "विश्लेषण पूर्ण झाले. पडताळता येणारे निष्कर्ष मिळाले नाहीत."],
  uploadWithheld: ["Analysis complete. {count} findings were withheld.", "विश्लेषण पूरा हुआ। {count} निष्कर्ष रोक दिए गए।", "विश्लेषण पूर्ण झाले. {count} निष्कर्ष रोखले आहेत."],
  uploadDone: ["Analysis complete. {count} findings passed the checks.", "विश्लेषण पूरा हुआ। {count} निष्कर्ष जाँच में सही पाए गए।", "विश्लेषण पूर्ण झाले. {count} निष्कर्ष तपासणीत योग्य आढळले."],
  uploadFailed: ["Analysis could not be completed. Try again or check the connection.", "विश्लेषण पूरा नहीं हो सका। फिर कोशिश करें या कनेक्शन जाँचें।", "विश्लेषण पूर्ण झाले नाही. पुन्हा प्रयत्न करा किंवा जोडणी तपासा."],
  uploadOverlayMismatch: ["Report ready. The video overlay may not align; see the technical details.", "रिपोर्ट तैयार है। वीडियो पर दिख रही आकृतियाँ सही जगह पर नहीं हो सकतीं; तकनीकी विवरण देखें।", "अहवाल तयार आहे. व्हिडिओवरील आकृत्या योग्य जागी दिसतीलच असे नाही; तांत्रिक तपशील पाहा."],
  uploadStageFailed: ["Analysis partly completed. Stage {agent} failed; see the technical details.", "विश्लेषण आंशिक रूप से पूरा हुआ। चरण {agent} विफल हुआ; तकनीकी विवरण देखें।", "विश्लेषण अंशतः पूर्ण झाले. टप्पा {agent} अयशस्वी झाला; तांत्रिक तपशील पाहा."],
  generating: ["Preparing report…", "रिपोर्ट तैयार हो रही है…", "अहवाल तयार होत आहे…"],
  fetching: ["Loading report…", "रिपोर्ट लोड हो रही है…", "अहवाल लोड होत आहे…"],
  simulating: ["Preparing sample days…", "नमूने के दिनों की जानकारी तैयार हो रही है…", "नमुन्यातील दिवसांची माहिती तयार होत आहे…"],
  writing: ["Writing report…", "रिपोर्ट लिखी जा रही है…", "अहवाल लिहिला जात आहे…"],
  generationFailed: ["The report could not be generated. Check the connection and try again.", "रिपोर्ट नहीं बन सकी। कनेक्शन जाँचें और फिर कोशिश करें।", "अहवाल तयार झाला नाही. जोडणी तपासा आणि पुन्हा प्रयत्न करा."],
  connectionReplay: ["Demo", "नमूना", "नमुना"],
  connectionLive: ["Connected", "जुड़ा है", "जोडले आहे"],
  connectionChecking: ["Connecting", "जोड़ रहे हैं", "जोडत आहे"],
  connectionLoading: ["Loading", "लोड हो रहा है", "लोड होत आहे"],
  connectionDown: ["Disconnected", "जुड़ा नहीं है", "जोडलेले नाही"],
  connectionReady: ["Connected. You can create a report or analyse a video.", "जुड़ गया है। अब रिपोर्ट बना सकते हैं या वीडियो का विश्लेषण कर सकते हैं।", "जोडले आहे. आता अहवाल तयार करता येईल किंवा व्हिडिओचे विश्लेषण करता येईल."],
  connectionWaiting: ["The analysis service is loading. This will update automatically.", "विश्लेषण सेवा लोड हो रही है। स्थिति अपने आप अपडेट होगी।", "विश्लेषण सेवा लोड होत आहे. स्थिती आपोआप अद्ययावत होईल."],
  connectionProbe: ["Checking the connection…", "कनेक्शन की जाँच हो रही है…", "जोडणी तपासत आहे…"],
  connectionError: ["The connection is unavailable. Check the address and try again.", "कनेक्शन उपलब्ध नहीं है। पता जाँचें और फिर कोशिश करें।", "जोडणी उपलब्ध नाही. पत्ता तपासा आणि पुन्हा प्रयत्न करा."],
  connectionDemo: ["Showing a sample report. Connect the local service for video analysis.", "नमूना रिपोर्ट दिखाई जा रही है। वीडियो विश्लेषण के लिए स्थानीय सेवा से जुड़ें।", "नमुना अहवाल दाखवत आहोत. व्हिडिओ विश्लेषणासाठी स्थानिक सेवेशी जोडा."],
  connectionNone: ["No backend connected", "कोई सेवा जुड़ी नहीं है", "कोणतीही सेवा जोडलेली नाही"],
  navReport: ["Report", "रिपोर्ट", "अहवाल"],
  navFootage: ["Upload a clip", "वीडियो अपलोड करें", "व्हिडिओ अपलोड करा"],
  navEvidence: ["Evidence checks", "प्रमाण की जाँच", "पुराव्यांची तपासणी"],
  navPipeline: ["How it works", "यह कैसे काम करता है", "हे कसे काम करते"],
  navResults: ["Results", "परिणाम", "निकाल"],
  brandSubtitle: ["Caregiver reports with checked evidence", "जाँचे गए प्रमाण के साथ देखभाल की रिपोर्ट", "तपासलेल्या पुराव्यांसह काळजीसाठी अहवाल"],
  reportEyebrow: ["Behaviour monitoring · research prototype", "व्यवहार निगरानी · शोध का प्रारूप", "वर्तन निरीक्षण · संशोधनाचा नमुना"],
  reportHeadline: ["A clearer daily report, with evidence you can check.", "दिन की स्पष्ट रिपोर्ट, ऐसे प्रमाण के साथ जिन्हें आप जाँच सकें।", "दिवसाचा स्पष्ट अहवाल, तुम्हाला तपासता येतील अशा पुराव्यांसह."],
  reportIntro: ["Read a caregiver summary alongside findings checked against recorded evidence. Five deterministic checks cover the structured claims. The model's summary and recommendation remain separate, unverified prose.", "देखभाल का सारांश और दर्ज प्रमाण से मिलान किए गए निष्कर्ष पढ़ें। पाँच तय नियमों वाली जाँच संरचित दावों पर लागू होती है। मॉडल का सारांश और सुझाव अलग हैं; उनकी जाँच नहीं होती।", "काळजीसाठीचा सारांश आणि नोंदवलेल्या पुराव्यांशी पडताळलेले निष्कर्ष वाचा. पाच निश्चित नियमांच्या तपासण्या संरचित दाव्यांना लागू होतात. मॉडेलचा सारांश आणि शिफारस स्वतंत्र आहेत; त्यांची पडताळणी होत नाही."],
  footageEyebrow: ["On your own footage", "अपने वीडियो पर", "आपल्या व्हिडिओवर"],
  footageHeading: ["Upload a clip to see the observations and caregiver report.", "निरीक्षण और देखभाल की रिपोर्ट देखने के लिए वीडियो अपलोड करें।", "निरीक्षणे आणि काळजीसाठीचा अहवाल पाहण्यासाठी व्हिडिओ अपलोड करा."],
  uploadIntro: ["Choose a video to see the recorded activities and a caregiver report. The report shows which findings passed the evidence checks.", "दर्ज गतिविधियाँ और देखभाल की रिपोर्ट देखने के लिए वीडियो चुनें। रिपोर्ट बताती है कि किन निष्कर्षों ने प्रमाण की जाँच पार की।", "नोंदवलेल्या हालचाली आणि काळजीसाठीचा अहवाल पाहण्यासाठी व्हिडिओ निवडा. कोणते निष्कर्ष पुराव्यांच्या तपासणीत योग्य ठरले ते अहवालात दिसते."],
  uploadCaution: ["Activity values come from this clip. Comparisons use a simulated reference, not this person's recorded daily history. A short clip cannot establish a personal daily decline.", "गतिविधियों के मान इस वीडियो से लिए गए हैं। तुलना काल्पनिक संदर्भ से है, इस व्यक्ति के दर्ज दैनिक इतिहास से नहीं। छोटे वीडियो से व्यक्ति की दैनिक स्थिति में गिरावट साबित नहीं होती।", "हालचालींची मोजमापे या व्हिडिओमधून घेतली आहेत. तुलना काल्पनिक संदर्भाशी आहे; या व्यक्तीच्या नोंदवलेल्या दैनंदिन इतिहासाशी नाही. छोट्या व्हिडिओवरून व्यक्तीच्या दैनंदिन स्थितीतील घसरण सिद्ध होत नाही."],
  uploadLimits: ["Up to 60 MB and 12,000 retained frames, about ten minutes at 20 Hz. Longer uploads are refused. Each tracked person's activity is shown separately.", "अधिकतम 60 MB और 12,000 रखे गए फ़्रेम, यानी 20 Hz पर लगभग दस मिनट। इससे लंबे अपलोड स्वीकार नहीं होते। हर व्यक्ति की गतिविधि अलग दिखाई जाती है।", "जास्तीत जास्त 60 MB आणि 12,000 ठेवलेल्या फ्रेम, म्हणजे 20 Hz वर सुमारे दहा मिनिटे. यापेक्षा मोठे अपलोड स्वीकारले जात नाहीत. प्रत्येक व्यक्तीच्या हालचाली स्वतंत्र दाखवल्या जातात."],
  enrolDisclosure: ["Optional enrollment saves the subject's name and an identity embedding on this machine. Saved names, roles and embeddings are sent to Kaggle for matching on later uploads. To forget enrollment, stop the bridge, delete its gallery file and restart it.", "वैकल्पिक पंजीकरण इस मशीन पर व्यक्ति का नाम और पहचान का संख्यात्मक प्रतिनिधित्व सहेजता है। बाद के अपलोड में पहचान मिलाने के लिए सहेजे गए नाम, भूमिकाएँ और प्रतिनिधित्व Kaggle को भेजे जाते हैं। पंजीकरण हटाने के लिए स्थानीय ब्रिज रोकें, उसकी गैलरी फ़ाइल मिटाएँ और उसे फिर शुरू करें।", "ऐच्छिक नोंदणीमुळे या संगणकावर व्यक्तीचे नाव आणि ओळखीचे संख्यात्मक प्रतिनिधित्व जतन होते. पुढील अपलोडमध्ये ओळख जुळवण्यासाठी जतन केलेली नावे, भूमिका आणि प्रतिनिधित्व Kaggle कडे पाठवले जातात. नोंदणी हटवण्यासाठी स्थानिक ब्रिज थांबवा, त्याची गॅलरी फाइल हटवा आणि तो पुन्हा सुरू करा."],
  backendHeading: ["Connect the inference backend", "विश्लेषण सेवा से जुड़ें", "विश्लेषण सेवेशी जोडा"],
  backendIntro: ["Connect to the local bridge at http://127.0.0.1:8899. It sends the clip to the Kaggle GPU and asks a hosted model to write from structured evidence. Reporting keys stay on the operator's machine.", "http://127.0.0.1:8899 पर स्थानीय ब्रिज से जुड़ें। यह वीडियो Kaggle GPU को भेजता है और बाहरी मॉडल से संरचित प्रमाण के आधार पर रिपोर्ट लिखवाता है। रिपोर्टिंग की API कुंजियाँ संचालक की मशीन पर रहती हैं।", "http://127.0.0.1:8899 वरील स्थानिक ब्रिजशी जोडा. तो व्हिडिओ Kaggle GPU कडे पाठवतो आणि बाह्य मॉडेलकडून संरचित पुराव्यांवर अहवाल लिहून घेतो. अहवालासाठीच्या API किल्ल्या संचालकाच्या संगणकावर राहतात."],
  backendLabel: ["Backend address", "सेवा का पता", "सेवेचा पत्ता"],
  tokenLabel: ["Token (optional)", "टोकन (वैकल्पिक)", "टोकन (ऐच्छिक)"],
  backendHelp: ["For setup, run notebook 05, then start web/local_backend.py with its tunnel address using --kaggle. Enter the local bridge address here. The optional token is for a direct connection to a protected notebook.", "सेटअप के लिए नोटबुक 05 चलाएँ, फिर उसके टनल पते को --kaggle में देकर web/local_backend.py शुरू करें। यहाँ स्थानीय ब्रिज का पता डालें। वैकल्पिक टोकन सुरक्षित नोटबुक से सीधे जुड़ने के लिए है।", "सेटअपसाठी नोटबुक 05 चालवा, नंतर त्याचा टनेल पत्ता --kaggle मध्ये देऊन web/local_backend.py सुरू करा. येथे स्थानिक ब्रिजचा पत्ता भरा. ऐच्छिक टोकन संरक्षित नोटबुकशी थेट जोडण्यासाठी आहे."],
  originalProseLabel: ["Original model summary and recommendation. These paragraphs are not verified by C1–C5.", "मॉडल का मूल सारांश और सुझाव। इन अनुच्छेदों की C1–C5 से जाँच नहीं होती।", "मॉडेलचा मूळ सारांश आणि शिफारस. या परिच्छेदांची C1–C5 द्वारे पडताळणी होत नाही."],
  claimChecksScope: ["Checks cover individual findings, not the summary or recommendation.", "जाँच अलग-अलग निष्कर्षों की होती है, सारांश या सुझाव की नहीं।", "तपासणी स्वतंत्र निष्कर्षांची होते; सारांश किंवा शिफारशीची नाही."],
  skipContent: ["Skip to the report", "रिपोर्ट पर जाएँ", "अहवालाकडे जा"],
  injectFaults: ["Demo faults", "नमूने की त्रुटियाँ", "नमुन्यातील चुका"],
  faultNone: ["None", "कोई नहीं", "एकही नाही"],
  faultSome: ["1 in 3", "3 में से 1", "3 पैकी 1"],
  faultAll: ["Every claim", "हर दावा", "प्रत्येक दावा"],
  caregiverTitle: ["Caregiver report", "देखभाल के लिए रिपोर्ट", "काळजीसाठी अहवाल"],
  dailyReport: ["Daily report", "दैनिक रिपोर्ट", "दैनंदिन अहवाल"],
  resident: ["Resident", "निवासी", "रहिवासी"],
  totalClaimsLabel: ["Claims", "दावे", "दावे"],
  shownClaimsLabel: ["Shown", "दिखाए गए", "दाखवलेले"],
  withheldClaimsLabel: ["Withheld", "शामिल नहीं किए", "वगळलेले"],
  summary: ["What was observed", "क्या देखा गया", "काय दिसून आले"],
  recommendation: ["Suggested next step", "सुझाया गया अगला कदम", "सुचवलेली पुढची कृती"],
  findings: ["Findings that passed the checks", "जाँच में सही पाए गए निष्कर्ष", "तपासणीत योग्य आढळलेले निष्कर्ष"],
  noSummary: ["No summary was provided.", "सारांश उपलब्ध नहीं है।", "सारांश दिलेला नाही."],
  noRecommendation: ["No recommendation was provided.", "कोई सुझाव उपलब्ध नहीं है।", "शिफारस दिलेली नाही."],
  translating: ["Translating the summary and recommendation…", "सारांश और सुझाव का अनुवाद हो रहा है…", "सारांश आणि शिफारशीचा अनुवाद होत आहे…"],
  translationUnavailable: ["The summary and recommendation could not be translated. Read the findings below or open the original English report.", "सारांश और सुझाव का अनुवाद नहीं हो सका। नीचे दिए निष्कर्ष पढ़ें या मूल अंग्रेज़ी रिपोर्ट खोलें।", "सारांश आणि शिफारशीचे भाषांतर झाले नाही. खालील निष्कर्ष वाचा किंवा मूळ इंग्रजी अहवाल उघडा."],
  retryTranslation: ["Retry translation", "अनुवाद फिर से करें", "भाषांतराचा पुन्हा प्रयत्न करा"],
  noFindings: ["No findings passed the checks. This does not establish that everything is well.", "कोई निष्कर्ष जाँच में सही नहीं पाया गया। इससे यह साबित नहीं होता कि सब ठीक है।", "एकही निष्कर्ष तपासणीत योग्य ठरला नाही. यावरून सर्व काही ठीक आहे असे ठरत नाही."],
  noScorable: ["There were no findings that could be checked. No conclusion can be drawn from this report.", "जाँचने योग्य निष्कर्ष नहीं थे। इस रिपोर्ट से कोई निष्कर्ष नहीं निकाला जा सकता।", "तपासता येतील असे निष्कर्ष नव्हते. या अहवालावरून निष्कर्ष काढता येत नाही."],
  counts: ["{accepted} findings shown · {withheld} withheld · {total} generated", "{accepted} निष्कर्ष दिखाए गए · {withheld} रोके गए · {total} बनाए गए", "{accepted} निष्कर्ष दाखवले · {withheld} रोखले · {total} तयार झाले"],
  escalateYes: ["The source report requests escalation. Review the recommendation and contact the appropriate caregiver.", "मूल रिपोर्ट आगे सहायता लेने को कहती है। सुझाव पढ़ें और उपयुक्त देखभालकर्ता से संपर्क करें।", "मूळ अहवाल पुढील मदत घेण्यास सांगतो. शिफारस वाचा आणि योग्य काळजीवाहू व्यक्तीशी संपर्क करा."],
  escalateNo: ["The source report does not request escalation. This is not a guarantee of safety.", "मूल रिपोर्ट आगे सहायता लेने को नहीं कहती। यह सुरक्षा की गारंटी नहीं है।", "मूळ अहवाल पुढील मदत घेण्यास सांगत नाही. ही सुरक्षिततेची हमी नाही."],
  escalateUnknown: ["No escalation decision was provided.", "आगे सहायता लेने का निर्णय उपलब्ध नहीं है।", "पुढील मदत घेण्याचा निर्णय दिलेला नाही."],
  simulatedCaution: ["The comparison uses a simulated reference, not this person's measured history. It cannot establish a clinical change.", "तुलना काल्पनिक संदर्भ से है, इस व्यक्ति के मापे गए इतिहास से नहीं। इससे स्वास्थ्य में बदलाव सिद्ध नहीं होता।", "तुलना काल्पनिक संदर्भाशी केली आहे; या व्यक्तीच्या मोजलेल्या इतिहासाशी नाही. यावरून आरोग्यातील बदल सिद्ध होत नाही."],
  demoCaution: ["This is a simulated sample report, not an observation of a real resident.", "यह काल्पनिक नमूना रिपोर्ट है, किसी वास्तविक निवासी का निरीक्षण नहीं।", "हा काल्पनिक नमुना अहवाल आहे; प्रत्यक्ष रहिवाशाचे निरीक्षण नाही."],
  clipCaution: ["This observation is too limited for a reliable daily comparison. Counts and durations describe only the observed window.", "रोज़ की विश्वसनीय तुलना के लिए यह निरीक्षण सीमित है। संख्या और अवधि केवल देखे गए समय की हैं।", "विश्वसनीय दैनंदिन तुलनेसाठी हे निरीक्षण मर्यादित आहे. संख्या आणि कालावधी फक्त पाहिलेल्या वेळेचे आहेत."],
  observedHours: ["Observed time: {hours} hours", "देखा गया समय: {hours} घंटे", "निरीक्षणाचा वेळ: {hours} तास"],
  reportDay: ["Report date: {day}", "रिपोर्ट की तारीख: {day}", "अहवालाची तारीख: {day}"],
  recordedOn: ["Recorded on {day}", "{day} को दर्ज किया गया", "{day} रोजी नोंदवले"],
  verificationScope: ["Checks assess the original structured findings against recorded evidence. They do not verify the summary, recommendation, translation, or medical accuracy.", "जाँच मूल संरचित निष्कर्षों का दर्ज प्रमाण से मिलान करती है। सारांश, सुझाव, अनुवाद या चिकित्सीय सटीकता की जाँच नहीं होती।", "तपासणी मूळ संरचित निष्कर्षांची नोंदवलेल्या पुराव्यांशी तुलना करते. सारांश, शिफारस, भाषांतर किंवा वैद्यकीय अचूकता तपासली जात नाही."],
  originalEnglish: ["Read the original English report", "मूल अंग्रेज़ी रिपोर्ट पढ़ें", "मूळ इंग्रजी अहवाल वाचा"],
  unknownFeature: ["Recorded measure", "दर्ज माप", "नोंदवलेले मोजमाप"],
  missingValue: ["No numeric value was stated.", "कोई संख्यात्मक मान नहीं बताया गया।", "संख्यात्मक मूल्य दिलेले नाही."],
  baseline: ["Reference: {value}.", "संदर्भ: {value}।", "संदर्भ: {value}."],
  change: ["Reported change: {value}%.", "बताया गया बदलाव: {value}%।", "नोंदवलेला बदल: {value}%."],
  direction_increase: ["Reported direction: increase.", "बताई गई दिशा: वृद्धि।", "नोंदवलेली दिशा: वाढ."],
  direction_decrease: ["Reported direction: decrease.", "बताई गई दिशा: कमी।", "नोंदवलेली दिशा: घट."],
  direction_unchanged: ["Reported direction: unchanged.", "बताई गई दिशा: अपरिवर्तित।", "नोंदवलेली दिशा: बदल नाही."],
  unit_seconds: ["seconds", "सेकंड", "सेकंद"],
  unit_hours: ["hours", "घंटे", "तास"],
  unit_count: ["events", "बार", "वेळा"],
  unit_people: ["people", "व्यक्ति", "व्यक्ती"],
  unit_score: ["points", "अंक", "गुण"],
  unit_ratio: ["(ratio, 0–1)", "(अनुपात, 0–1)", "(गुणोत्तर, 0–1)"],
};

const features = {
  walking_duration_s: ["Time spent walking", "चलने में बिताया समय", "चालण्यात घालवलेला वेळ"],
  walking_bouts: ["Walking periods", "चलने के दौर", "चालण्याचे टप्पे"],
  mean_bout_duration_s: ["Average walking period", "चलने के दौर की औसत अवधि", "चालण्याच्या टप्प्याचा सरासरी कालावधी"],
  room_transitions: ["Movement between rooms", "कमरों के बीच आवाजाही", "खोल्यांमधील ये-जा"],
  sit_to_stand_count: ["Times rising from a seat", "बैठने से उठने की संख्या", "बसलेल्या स्थितीतून उठण्याची संख्या"],
  mean_sit_to_stand_duration_s: ["Average time to stand up", "खड़े होने का औसत समय", "उभे राहण्यासाठीचा सरासरी वेळ"],
  sitting_duration_s: ["Time spent sitting", "बैठने में बिताया समय", "बसण्यात घालवलेला वेळ"],
  lying_duration_s: ["Time spent lying down", "लेटने में बिताया समय", "आडवे पडण्यात घालवलेला वेळ"],
  standing_duration_s: ["Time spent standing", "खड़े रहने में बिताया समय", "उभे राहण्यात घालवलेला वेळ"],
  longest_inactive_block_s: ["Longest inactive period", "बिना गतिविधि का सबसे लंबा समय", "हालचाल नसलेला सर्वात मोठा कालावधी"],
  meal_events: ["Meals recorded", "दर्ज भोजन की संख्या", "नोंदवलेली जेवणे"],
  eating_duration_s: ["Time spent eating", "खाने में बिताया समय", "जेवण्यात घालवलेला वेळ"],
  drinking_events: ["Drinking events", "पीने की संख्या", "पिण्याच्या नोंदी"],
  cooking_duration_s: ["Time spent preparing food", "खाना बनाने में बिताया समय", "स्वयंपाकात घालवलेला वेळ"],
  medication_events: ["Medication events", "दवा लेने की संख्या", "औषध घेण्याच्या नोंदी"],
  tv_duration_s: ["Time spent watching television", "टीवी देखने में बिताया समय", "टीव्ही पाहण्यात घालवलेला वेळ"],
  reading_duration_s: ["Time spent reading", "पढ़ने में बिताया समय", "वाचनात घालवलेला वेळ"],
  phone_events: ["Phone use events", "फ़ोन इस्तेमाल की संख्या", "फोन वापरण्याच्या नोंदी"],
  social_interaction_duration_s: ["Time in social interaction", "सामाजिक बातचीत में बिताया समय", "सामाजिक संवादात घालवलेला वेळ"],
  visitor_count: ["Visitors recorded", "दर्ज आगंतुकों की संख्या", "नोंदवलेले पाहुणे"],
  housework_duration_s: ["Time spent on housework", "घर के काम में बिताया समय", "घरकामात घालवलेला वेळ"],
  fall_events: ["Falls recorded", "दर्ज गिरने की संख्या", "नोंदवलेले पडण्याचे प्रसंग"],
  max_post_fall_immobility_s: ["Longest immobility after a fall", "गिरने के बाद बिना हिले सबसे लंबा समय", "पडल्यानंतर हालचाल नसलेला सर्वात मोठा कालावधी"],
  observed_hours: ["Observed time", "देखा गया समय", "निरीक्षणाचा वेळ"],
  tracking_coverage: ["Person tracking coverage", "व्यक्ति का पता लगने वाले समय का अनुपात", "व्यक्तीचा मागोवा घेतलेल्या वेळेचे प्रमाण"],
  mobility_index: ["Mobility score", "चलने-फिरने के अंक", "हालचालींचे गुण"],
  sedentary_ratio: ["Share of posture time sitting or lying", "बैठने या लेटने में बिताए समय का अनुपात", "बसण्यात किंवा आडवे पडण्यात घालवलेल्या वेळेचे प्रमाण"],
};
for (const [key, value] of Object.entries(features)) rows[`feature_${key}`] = value;

export function getLanguage() { return language; }
export function t(key, params = {}, selected = language) {
  const column = { en: 0, hi: 1, mr: 2 }[selected] ?? 0;
  const value = rows[key]?.[column] ?? rows[key]?.[0] ?? key;
  return value.replace(/\{(\w+)\}/g, (match, name) => Object.hasOwn(params, name) ? String(params[name]) : match);
}

function translateNodes() {
  if (typeof document === "undefined") return;
  document.querySelectorAll("[data-i18n]").forEach((node) => {
    node.textContent = t(node.dataset.i18n);
    node.lang = language;
  });
  document.querySelectorAll("[data-i18n-placeholder]").forEach((node) => {
    node.setAttribute("placeholder", t(node.dataset.i18nPlaceholder));
    node.lang = language;
  });
  if (boundSelect) boundSelect.value = language;
}

export function setLanguage(code) {
  if (!Object.hasOwn(LANGUAGES, code)) return false;
  const changed = language !== code;
  language = code;
  try { globalThis.localStorage?.setItem(STORAGE_KEY, code); } catch { /* Disabled storage is fine. */ }
  translateNodes();
  if (changed) for (const listener of listeners) listener(code);
  return true;
}

export function onLanguageChange(fn) {
  listeners.add(fn);
  return () => listeners.delete(fn);
}

export function initI18n() {
  try {
    const saved = globalThis.localStorage?.getItem(STORAGE_KEY);
    if (Object.hasOwn(LANGUAGES, saved)) language = saved;
  } catch { /* Remains usable without persisted preferences. */ }
  if (typeof document !== "undefined") {
    const select = document.getElementById("report-language");
    if (select && select !== boundSelect) {
      boundSelect = select;
      select.replaceChildren(...Object.entries(LANGUAGES).map(([code, label]) => {
        const option = document.createElement("option");
        option.value = code; option.textContent = label; option.lang = code;
        return option;
      }));
      select.addEventListener("change", () => setLanguage(select.value));
    }
  }
  translateNodes();
  return language;
}
