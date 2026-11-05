# E new-output human review order

Ordering is based only on frozen metrics and whether the answer changed, and does not represent a human conclusion. 128 results, 123 unique dataset/sample_id; each batch is kept separately.

Review the first 42 changes first (17 metric regressions, 4 metric improvements, 21 other changes); the remaining 86 results that kept the answer still await full evidence-support review.

For the same item, the human reviewer must judge the initial answer and the final answer separately on correctness and evidence support, four items in total; record the reviewer and the rationale.
Write confirmations into pair_review in review_priority.jsonl, and set status=completed when done; the four boolean fields must not be empty.
The original pending_semantic_pairs.jsonl is kept as an immutable audit base record. Do not use the frozen references to auto-fill human fields.

Existing correctness judgments must also be rechecked under the same criteria. Citations taken verbatim from the source text only prove the source is legitimate; returning the original answer does not mean the answer has been verified.

## 1. FEVER / dev50 / 140820

Priority reason: frozen_metric_regression; result fingerprint: `5017ca4bd2f346b8d66663c832be083adff27c902ac1b212313375382879e15b`.

Question: Men in Black II is a film from America.

Initial: SUPPORTS

Final: REFUTES

Human status: pending; neither correctness nor support has been judged.

## 2. FEVER / dev50 / 33982

Priority reason: frozen_metric_regression; result fingerprint: `a465787cdacf4a3c272cc098daf71dc2c96b5e0d6e4cfe24625e27b019935f4c`.

Question: Henry Cavill starred in two Superman films.

Initial: SUPPORTS

Final: REFUTES

Human status: pending; neither correctness nor support has been judged.

## 3. NQ / reviewed / 5259818422556035007

Priority reason: frozen_metric_regression; result fingerprint: `11d506c9044436059e2bb23850ab27f9b8318392c6f90b48169a71b1344349d3`.

Question: who sings good night on the beatles white album

Initial: Ringo Starr

Final: It is sung by Ringo Starr , the only Beatle to appear on the track .

Human status: pending; neither correctness nor support has been judged.

## 4. NQ / reviewed / 970104939222709777

Priority reason: frozen_metric_regression; result fingerprint: `279dd6c9369dc81eabb71fa882d757bf3c7f25f0d148fd479710da21d6577d35`.

Question: who is credited with creating the gothic art movement

Initial: Abbot Suger

Final: Gothic art emerged in Île - de-France , France , in the early 12th century at the Abbey Church of St Denis built by Abbot Suger .

Human status: pending; neither correctness nor support has been judged.

## 5. NQ / dev50 / -1450346568093066821

Priority reason: frozen_metric_regression; result fingerprint: `b47707d53914b86ac444f1dd69fa52325c4aa472aebadcde07ebee0ec4fb7531`.

Question: where was because i could not stop for death published

Initial: Poems: Series 1

Final: Because I could not stop for Death '' is a lyrical poem by Emily Dickinson first published posthumously in Poems : Series 1 in 1890 .

Human status: pending; neither correctness nor support has been judged.

## 6. NQ / dev50 / -2680244887428298888

Priority reason: frozen_metric_regression; result fingerprint: `998669c4a1fe4c33eb21c212825710e263ea57ad3cd6d3b93939f3364869e26b`.

Question: how many lines of symmetry are there in a equilateral triangle

Initial: 3

Final: An equilateral triangle is the most symmetrical triangle , having 3 lines of reflection and rotational symmetry of order 3 about its center .

Human status: pending; neither correctness nor support has been judged.

## 7. NQ / dev50 / -3064086919220674644

Priority reason: frozen_metric_regression; result fingerprint: `816314b440f42e262c4e22384313287daa83304131fec48ad61228a27ca5f63c`.

Question: when did the great plague of london start and end

Initial: 1665 to 1666

Final: The Great Plague , lasting from 1665 to 1666 , was the last major epidemic of the bubonic plague to occur in England .

Human status: pending; neither correctness nor support has been judged.

## 8. NQ / dev50 / -3339006116507262453

Priority reason: frozen_metric_regression; result fingerprint: `1464ebe8e1bab6273dd22b9b44b697e6e3e7f7e68f4b70646562b32e6d3fa566`.

Question: who sang i ran all the way home

Initial: The Impalas

Final: The Impalas were an American doo - wop group in the late 1950s , best known for their hit , `` Sorry ( I Ran All the Way Home ) '' .

Human status: pending; neither correctness nor support has been judged.

## 9. NQ / dev50 / -5748368670037082784

Priority reason: frozen_metric_regression; result fingerprint: `35422eac193e4faf37b37021d60baa483c1b895e0064396ea315a10a54641f48`.

Question: what is the definition of the name thomas

Initial: twin

Final: Thomas was a popular medieval given throughout Europe , coming from the biblical name , Thomas being one of Jesus ' disciples . It is ultimately derived from the Aramaic personal name meaning `` twin . ''

Human status: pending; neither correctness nor support has been judged.

## 10. NQ / dev50 / -6089464774983144695

Priority reason: frozen_metric_regression; result fingerprint: `1f7bcb98f0303333d0897d235bf635c6c01ab6349e325cf293e5a69dd9271890`.

Question: who plays grace in the secret life of the american teenager

Initial: Megan Park

Final: Megan Park ( born July 24 , 1986 ) is a Canadian actress and singer . She is known for her portrayal of Grace Bowman in the television series The Secret Life of the American Teenager .

Human status: pending; neither correctness nor support has been judged.

## 11. NQ / dev50 / -8209571805051987276

Priority reason: frozen_metric_regression; result fingerprint: `b62d234ca60df73d791546348485b4ae9c0f419358926ad29c5d9c8c6cc0d53d`.

Question: what is the genus of a bald eagle

Initial: Haliaeetus

Final: The bald eagle placed in the genus Haliaeetus

Human status: pending; neither correctness nor support has been judged.

## 12. NQ / dev50 / -8631190913794477985

Priority reason: frozen_metric_regression; result fingerprint: `a075bf0b926cabce1459515094c24c7107b413c1d4fe9cf54d0c06de638809e2`.

Question: who scored the most points in a single game in the nba

Initial: Wilt Chamberlain

Final: Chamberlain holds the single - game scoring record , having scored 100 in game in 1962 .

Human status: pending; neither correctness nor support has been judged.

## 13. NQ / dev50 / -8825841420882619284

Priority reason: frozen_metric_regression; result fingerprint: `667c261b602fc98769c83ff9603a5a3f6abf7c1862c25a2aacbd4f88436e5194`.

Question: where's the tv show the crossing filmed

Initial: British Columbia, Canada

Final: Set in the fictional town of Port Canaan , Oregon and in Seattle , the series was filmed in coastal areas of British Columbia and in the city of Vancouver , British Columbia in 2017 .

Human status: pending; neither correctness nor support has been judged.

## 14. NQ / dev50 / 1117400623925969758

Priority reason: frozen_metric_regression; result fingerprint: `d9c89e96fa33470f213618918e7ce3a922e855504f3601f5fed8a80af7d4923e`.

Question: who won the ncaa women's championship last year 2017

Initial: South Carolina

Final: South Carolina defeated Mississippi State to win the championship .

Human status: pending; neither correctness nor support has been judged.

## 15. NQ / dev50 / 2435796066046322163

Priority reason: frozen_metric_regression; result fingerprint: `018c65e3b1705b13eaee9be077177e683c536c2ca12229ec56ac37db7d2eae41`.

Question: who donates a christmas tree to trafalgar square

Initial: the city of Oslo, Norway

Final: The Trafalgar Square Christmas tree is a Christmas tree donated to the people of Britain by the city of Oslo , Norway each year since 1947 .

Human status: pending; neither correctness nor support has been judged.

## 16. NQ / dev50 / 5898512601992001509

Priority reason: frozen_metric_regression; result fingerprint: `68d51555d6d65857560e183e1aad485b0a20fad97c8d5ad48e2a721f717234b2`.

Question: who did the united states fight in the war of 1812

Initial: the United Kingdom

Final: The War of 1812 ( 1812 -- 1815 ) was a conflict fought between the United States , the United Kingdom , and their respective allies .

Human status: pending; neither correctness nor support has been judged.

## 17. NQ / dev50 / 8727688733894535817

Priority reason: frozen_metric_regression; result fingerprint: `c0828c85a59a90f6f046f8912ce6f2b80400bc2110095985ef08e005ea353c70`.

Question: what was the purpose of the bay of pigs mission

Initial: to overthrow the increasingly communist government of Fidel Castro

Final: A counter-revolutionary military group ( made up of mostly Cuban exiles who traveled to the United States after Castro 's takeover , but also of some US military personnel ) , trained and funded by the CIA , Brigade 2506 fronted the armed wing of the Democratic Revolutionary Front ( DRF ) and intended to overthrow the increasingly communist government of Fidel Castro .

Human status: pending; neither correctness nor support has been judged.

## 18. FEVER / reviewed / 156493

Priority reason: frozen_metric_gain; result fingerprint: `b6b6e32b49cfad6b55a9eda781d1ca1bde34644e0b3f81290aabfbc42c74ab9a`.

Question: Daggering is not a type of dance.

Initial: SUPPORTS

Final: REFUTES

Human status: pending; neither correctness nor support has been judged.

## 19. FEVER / reviewed / 203615

Priority reason: frozen_metric_gain; result fingerprint: `820a36ef266d7747232e0ddad2984a2d23b451df2224b085c0682b20c6d3e2c3`.

Question: The 1974 crime film The Sugarland Express was co-written by Steven Spielberg.

Initial: REFUTES

Final: SUPPORTS

Human status: pending; neither correctness nor support has been judged.

## 20. FEVER / reviewed / 72578

Priority reason: frozen_metric_gain; result fingerprint: `9ebfa78b26e1c17fd8511d7d666c6c1ef205509c5af278884fe49abeb1e54c73`.

Question: The Greek language has speakers in Albania.

Initial: REFUTES

Final: SUPPORTS

Human status: pending; neither correctness nor support has been judged.

## 21. FEVER / dev50 / 203615

Priority reason: frozen_metric_gain; result fingerprint: `55181ab06bf23aeb3f632175451f1a4a990a2f0ca6954fb109bb3cc4bf1574d0`.

Question: The 1974 crime film The Sugarland Express was co-written by Steven Spielberg.

Initial: REFUTES

Final: SUPPORTS

Human status: pending; neither correctness nor support has been judged.

## 22. NQ / reviewed / -2282501421613384225

Priority reason: changed_without_frozen_correctness_gain; result fingerprint: `89a22356f9f6e273d5aad34fe2a8849566ea3ca336a1bd7cc7cdc4ca4e77913d`.

Question: where is the tibia and fibula bone located

Initial: on the lateral side

Final: The fibula or calf bone is a leg bone located on the lateral side of the tibia , with which it is connected above and below .

Human status: pending; neither correctness nor support has been judged.

## 23. NQ / reviewed / -2905696611778901658

Priority reason: changed_without_frozen_correctness_gain; result fingerprint: `7ab4cf0d320f4cd9e70359f19a53c1026174ffea7a1ca4011fa086a8ae7eb08e`.

Question: who sang oh annie i not your daddy

Initial: Annie, I'm Not Your Daddy

Final: Annie , I 'm Not Your Daddy is a song written by August Darnell and first recorded by his band Kid Creole and the Coconuts .

Human status: pending; neither correctness nor support has been judged.

## 24. NQ / reviewed / 2326758035558088175

Priority reason: changed_without_frozen_correctness_gain; result fingerprint: `b993917b44324598c7b1984d411f44c4335495b9920052f7d844f51ed97c85a4`.

Question: what is the meaning of to have a bias

Initial: prejudice in favor of or against one thing, person, or group compared with another

Final: Bias is prejudice in favour of or against one thing , person , or group compared with another , usually in a way considered to be unfair .

Human status: pending; neither correctness nor support has been judged.

## 25. NQ / reviewed / 2393389448844632496

Priority reason: changed_without_frozen_correctness_gain; result fingerprint: `237a814782400a05a1567c2bab882e2e6ae78f7676087a69b699eec32a66a1a8`.

Question: what is loihi and where does it originate from

Initial: Lōihi Seamount

Final: Lōihi Seamount ( also known as Lōʻihi ) is an active submarine volcano about 35 km ( 22 mi ) off the southeast coast of the island of Hawaii .

Human status: pending; neither correctness nor support has been judged.

## 26. NQ / reviewed / 4534870872535404381

Priority reason: changed_without_frozen_correctness_gain; result fingerprint: `7b9a42a3da86b1625536c672fb8e0d5bf4647a53deaca734363cf3cda82dc447`.

Question: what went wrong on the columbia space shuttle

Initial: foam insulation broke off from the Space Shuttle external tank and damaged the left wing

Final: During the launch of STS - 107 , Columbia 's 28th mission , a piece of foam insulation broke off from the Space Shuttle external tank and struck the left wing of the orbiter . A few previous shuttle launches had seen damage ranging from minor to major from foam shedding , but some engineers suspected that the damage to Columbia was more serious .

Human status: pending; neither correctness nor support has been judged.

## 27. NQ / reviewed / 6112920825267803506

Priority reason: changed_without_frozen_correctness_gain; result fingerprint: `b0fe9424e432dd4d1b3ceda25ff99c833adcd5cfe658c6657560eff8c2988759`.

Question: what is a serrated adenoma of the colon

Initial: sessile serrated adenoma (SSA)

Final: In gastroenterology , a sessile serrated adenoma ( abbreviated SSA ) , also known as sessile serrated polyp ( abbreviated SSP ) , is a premalignant flat ( or sessile ) lesion of the colon , predominantly seen in the cecum and ascending colon .

Human status: pending; neither correctness nor support has been judged.

## 28. NQ / reviewed / 7013168592046328137

Priority reason: changed_without_frozen_correctness_gain; result fingerprint: `5830021a4c8985c3a786f4a7cc4efb05f67ae68cf5f6dc909f3c939b54f665b7`.

Question: who did the astros play last time they were in the world series

Initial: Chicago White Sox

Final: The 2017 World Series took place between the Los Angeles Dodgers and Houston Astros .

Human status: pending; neither correctness nor support has been judged.

## 29. NQ / reviewed / 7446684003786592114

Priority reason: changed_without_frozen_correctness_gain; result fingerprint: `cc47f6d8620f209e59bf850fd1298740a96ebd406f68ccd9802e769f930b6974`.

Question: when does jenny come back to gossip girl

Initial: October 16, 2012

Final: on October 16 , 2012 , it was reported that Momsen would be returning to Gossip Girl for the final episode of the sixth and final season , in which she appeared briefly in the wedding sequence .

Human status: pending; neither correctness nor support has been judged.

## 30. NQ / dev50 / -1209334507323554133

Priority reason: changed_without_frozen_correctness_gain; result fingerprint: `4d4523ac5c6e6f5306a52265e1f3970ed717a1f968b68579cd62bc39af3fe776`.

Question: what degree is a crock pot on low

Initial: not specified

Final: Because the heating elements are generally located at the bottom and often also partway up the sides , most slow cookers have a minimum recommended liquid level to avoid uncontrolled heating .

Human status: pending; neither correctness nor support has been judged.

## 31. NQ / dev50 / -1680848497536972887

Priority reason: changed_without_frozen_correctness_gain; result fingerprint: `135391a819505cb617f23aa99f7670bcf155aa087a640216672c309386d89406`.

Question: what is property of a system in thermodynamics

Initial: any property that describes a state of a physical system

Final: Within thermodynamics , a physical property is any property that is measurable , and whose value describes a state of a physical system .

Human status: pending; neither correctness nor support has been judged.

## 32. NQ / dev50 / -1814654291537702625

Priority reason: changed_without_frozen_correctness_gain; result fingerprint: `9ff30adfe7c529476c1a03b5389341c767a95c1397473ce48514ede96b42d5c0`.

Question: what is the basic principle of christian morality

Initial: love

Final: The New Testament generally asserts that all morality flows from the Great Commandment , to love God with all one 's heart , mind , strength , and soul , and to love one 's neighbour as oneself .

Human status: pending; neither correctness nor support has been judged.

## 33. NQ / dev50 / -2282501421613384225

Priority reason: changed_without_frozen_correctness_gain; result fingerprint: `93117f02576eb6e2348325756e8e028b0b036451ea3d7d3387e45b9cec6070d1`.

Question: where is the tibia and fibula bone located

Initial: on the lateral side

Final: The fibula or calf bone is a leg bone located on the lateral side of the tibia , with which it is connected above and below .

Human status: pending; neither correctness nor support has been judged.

## 34. NQ / dev50 / -2743701347497238377

Priority reason: changed_without_frozen_correctness_gain; result fingerprint: `898d964112f2229dcdc5445f470088c8b904eaf87a22ef2414c2b6505f9942c3`.

Question: how oxygenated blood returns to the heart from the lungs

Initial: pulmonary veins

Final: The oxygenated blood then leaves the lungs through pulmonary veins , which return it to the left heart , completing the pulmonary cycle .

Human status: pending; neither correctness nor support has been judged.

## 35. NQ / dev50 / -2905696611778901658

Priority reason: changed_without_frozen_correctness_gain; result fingerprint: `7a236bf83bd5dbcad8b7fe6a0cd286e527defb51a2d5bad2aa317a74e6efb9d1`.

Question: who sang oh annie i not your daddy

Initial: Annie, I'm Not Your Daddy

Final: Annie , I 'm Not Your Daddy is a song written by August Darnell and first recorded by his band Kid Creole and the Coconuts .

Human status: pending; neither correctness nor support has been judged.

## 36. NQ / dev50 / -3089533754017582743

Priority reason: changed_without_frozen_correctness_gain; result fingerprint: `d3cb3d2822d82372b820e8aa98963d38d8a09b90012aeb3921040e5e311ac596`.

Question: when was the last time unc did not make the ncaa tournament

Initial: 2001

Final: The state of North Carolina was threatened with a 2018 - 2022 championship venue boycott by the NCAA , due to the HB2 law passed in 2016 .

Human status: pending; neither correctness nor support has been judged.

## 37. NQ / dev50 / -5973003791441271262

Priority reason: changed_without_frozen_correctness_gain; result fingerprint: `4059b317bd0b75c897c27f17c135c966a0329a045ff017350bb6280b16c69f9b`.

Question: where does the last name andersen originate from

Initial: Danish-Norwegian

Final: Andersen ( Danish pronunciation : ( ˈɑnɐsn̩ ) ) is a Danish - Norwegian patronymic surname meaning `` son of Anders '' ( itself derived from the Greek name `` Ανδρέας / Andreas '' , cf . English Andrew ) .

Human status: pending; neither correctness nor support has been judged.

## 38. NQ / dev50 / 2393389448844632496

Priority reason: changed_without_frozen_correctness_gain; result fingerprint: `1f7247f570ef8a83c119d9a4c445f0cf7f30e0226a1834fd1ddf40fda345b1c2`.

Question: what is loihi and where does it originate from

Initial: Lōihi Seamount

Final: Lōihi Seamount ( also known as Lōʻihi ) is an active submarine volcano about 35 km ( 22 mi ) off the southeast coast of the island of Hawaii .

Human status: pending; neither correctness nor support has been judged.

## 39. NQ / dev50 / 6781489363263681689

Priority reason: changed_without_frozen_correctness_gain; result fingerprint: `f63ccc82794a7a6b8219a2f135de8de5d1a2c737180647b43faf0caef6966a07`.

Question: where does the last name rudolph come from

Initial: Germanic name

Final: Rudolph or Rudolf ( French : Rodolphe , Italian , Portuguese and Spanish : Rodolfo ) or Rodolphe is a male first name , and , less commonly , a surname . It is a Germanic name deriving from two stems : Rod or Hrōð , meaning `` fame '' , and olf meaning `` wolf '' ( see also Hroðulf ; cf . Adolf ) .

Human status: pending; neither correctness nor support has been judged.

## 40. NQ / dev50 / 7632941217874741929

Priority reason: changed_without_frozen_correctness_gain; result fingerprint: `ae59ee0305a75755654f660db9637bf809c6fab89003bab945a2dcf685f75a2c`.

Question: what is george harrison song wah wah about

Initial: frustration with the atmosphere in the group

Final: The lyrics reflect his frustration with the atmosphere in the group at that time -- namely , Paul McCartney 's over-assertiveness and criticism of his guitar playing , John Lennon 's lack of engagement with the project and dismissal of Harrison as a songwriter , and Yoko Ono 's constant involvement in the band 's activities .

Human status: pending; neither correctness nor support has been judged.

## 41. NQ / dev50 / 8003762008471695244

Priority reason: changed_without_frozen_correctness_gain; result fingerprint: `3b60f197bd75588cdea75702fb04d870872ad92eba5ef2996194bea32dd4149e`.

Question: the christian church used which holiday to take attention away from samhain

Initial: Christmas

Final: It was a custom of the Pagans to celebrate on the same 25 December the birthday of the Sun , at which they kindled lights in token of festivity . In these solemnities and revelries the Christians also took part . Accordingly when the doctors of the Church perceived that the Christians had a leaning to this festival , they took counsel and resolved that the true Nativity should be solemnised on that day .

Human status: pending; neither correctness nor support has been judged.

## 42. NQ / dev50 / 970985133701043298

Priority reason: changed_without_frozen_correctness_gain; result fingerprint: `b35ada7901232f1c565b267f02615ce7e52450f52e4a62af9544a19714ae36b7`.

Question: a legislative act passed by congress is an example of

Initial: statute

Final: The act was passed by large majorities in both houses of Congress .

Human status: pending; neither correctness nor support has been judged.

