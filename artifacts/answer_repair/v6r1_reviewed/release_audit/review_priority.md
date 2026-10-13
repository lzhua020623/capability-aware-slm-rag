# E 新输出人工审核顺序

排序仅依据冻结指标和是否改答案，不代表人工结论。128 条结果，123 个唯一 dataset/sample_id；各批分别保留。

优先审核前 42 条改动（17 条指标退步、4 条指标改善、21 条其他改动）；其余 86 条保留答案的结果仍待完整证据支持审核。

人工需要对同一条的初始答案与最终答案分别判定：正确性、证据支持，共四项；记录审核人和理由。
请把确认写入 review_priority.jsonl 的 pair_review，完成时设 status=completed；四个布尔字段不能为空。
原始 pending_semantic_pairs.jsonl 保留为不可改写的审计底稿。不要用冻结参考自动填人工字段。

已有正确性判断也要按同一口径复查。引用字面来自原文，只能证明来源合法；返回原答案不代表答案已获证实。

## 1. FEVER / dev50 / 140820

优先原因：frozen_metric_regression；结果指纹：`5017ca4bd2f346b8d66663c832be083adff27c902ac1b212313375382879e15b`。

问题：Men in Black II is a film from America.

初始：SUPPORTS

最终：REFUTES

人工状态：pending，正确性与支持性均未判定。

## 2. FEVER / dev50 / 33982

优先原因：frozen_metric_regression；结果指纹：`a465787cdacf4a3c272cc098daf71dc2c96b5e0d6e4cfe24625e27b019935f4c`。

问题：Henry Cavill starred in two Superman films.

初始：SUPPORTS

最终：REFUTES

人工状态：pending，正确性与支持性均未判定。

## 3. NQ / reviewed / 5259818422556035007

优先原因：frozen_metric_regression；结果指纹：`11d506c9044436059e2bb23850ab27f9b8318392c6f90b48169a71b1344349d3`。

问题：who sings good night on the beatles white album

初始：Ringo Starr

最终：It is sung by Ringo Starr , the only Beatle to appear on the track .

人工状态：pending，正确性与支持性均未判定。

## 4. NQ / reviewed / 970104939222709777

优先原因：frozen_metric_regression；结果指纹：`279dd6c9369dc81eabb71fa882d757bf3c7f25f0d148fd479710da21d6577d35`。

问题：who is credited with creating the gothic art movement

初始：Abbot Suger

最终：Gothic art emerged in Île - de-France , France , in the early 12th century at the Abbey Church of St Denis built by Abbot Suger .

人工状态：pending，正确性与支持性均未判定。

## 5. NQ / dev50 / -1450346568093066821

优先原因：frozen_metric_regression；结果指纹：`b47707d53914b86ac444f1dd69fa52325c4aa472aebadcde07ebee0ec4fb7531`。

问题：where was because i could not stop for death published

初始：Poems: Series 1

最终：Because I could not stop for Death '' is a lyrical poem by Emily Dickinson first published posthumously in Poems : Series 1 in 1890 .

人工状态：pending，正确性与支持性均未判定。

## 6. NQ / dev50 / -2680244887428298888

优先原因：frozen_metric_regression；结果指纹：`998669c4a1fe4c33eb21c212825710e263ea57ad3cd6d3b93939f3364869e26b`。

问题：how many lines of symmetry are there in a equilateral triangle

初始：3

最终：An equilateral triangle is the most symmetrical triangle , having 3 lines of reflection and rotational symmetry of order 3 about its center .

人工状态：pending，正确性与支持性均未判定。

## 7. NQ / dev50 / -3064086919220674644

优先原因：frozen_metric_regression；结果指纹：`816314b440f42e262c4e22384313287daa83304131fec48ad61228a27ca5f63c`。

问题：when did the great plague of london start and end

初始：1665 to 1666

最终：The Great Plague , lasting from 1665 to 1666 , was the last major epidemic of the bubonic plague to occur in England .

人工状态：pending，正确性与支持性均未判定。

## 8. NQ / dev50 / -3339006116507262453

优先原因：frozen_metric_regression；结果指纹：`1464ebe8e1bab6273dd22b9b44b697e6e3e7f7e68f4b70646562b32e6d3fa566`。

问题：who sang i ran all the way home

初始：The Impalas

最终：The Impalas were an American doo - wop group in the late 1950s , best known for their hit , `` Sorry ( I Ran All the Way Home ) '' .

人工状态：pending，正确性与支持性均未判定。

## 9. NQ / dev50 / -5748368670037082784

优先原因：frozen_metric_regression；结果指纹：`35422eac193e4faf37b37021d60baa483c1b895e0064396ea315a10a54641f48`。

问题：what is the definition of the name thomas

初始：twin

最终：Thomas was a popular medieval given throughout Europe , coming from the biblical name , Thomas being one of Jesus ' disciples . It is ultimately derived from the Aramaic personal name meaning `` twin . ''

人工状态：pending，正确性与支持性均未判定。

## 10. NQ / dev50 / -6089464774983144695

优先原因：frozen_metric_regression；结果指纹：`1f7bcb98f0303333d0897d235bf635c6c01ab6349e325cf293e5a69dd9271890`。

问题：who plays grace in the secret life of the american teenager

初始：Megan Park

最终：Megan Park ( born July 24 , 1986 ) is a Canadian actress and singer . She is known for her portrayal of Grace Bowman in the television series The Secret Life of the American Teenager .

人工状态：pending，正确性与支持性均未判定。

## 11. NQ / dev50 / -8209571805051987276

优先原因：frozen_metric_regression；结果指纹：`b62d234ca60df73d791546348485b4ae9c0f419358926ad29c5d9c8c6cc0d53d`。

问题：what is the genus of a bald eagle

初始：Haliaeetus

最终：The bald eagle placed in the genus Haliaeetus

人工状态：pending，正确性与支持性均未判定。

## 12. NQ / dev50 / -8631190913794477985

优先原因：frozen_metric_regression；结果指纹：`a075bf0b926cabce1459515094c24c7107b413c1d4fe9cf54d0c06de638809e2`。

问题：who scored the most points in a single game in the nba

初始：Wilt Chamberlain

最终：Chamberlain holds the single - game scoring record , having scored 100 in game in 1962 .

人工状态：pending，正确性与支持性均未判定。

## 13. NQ / dev50 / -8825841420882619284

优先原因：frozen_metric_regression；结果指纹：`667c261b602fc98769c83ff9603a5a3f6abf7c1862c25a2aacbd4f88436e5194`。

问题：where's the tv show the crossing filmed

初始：British Columbia, Canada

最终：Set in the fictional town of Port Canaan , Oregon and in Seattle , the series was filmed in coastal areas of British Columbia and in the city of Vancouver , British Columbia in 2017 .

人工状态：pending，正确性与支持性均未判定。

## 14. NQ / dev50 / 1117400623925969758

优先原因：frozen_metric_regression；结果指纹：`d9c89e96fa33470f213618918e7ce3a922e855504f3601f5fed8a80af7d4923e`。

问题：who won the ncaa women's championship last year 2017

初始：South Carolina

最终：South Carolina defeated Mississippi State to win the championship .

人工状态：pending，正确性与支持性均未判定。

## 15. NQ / dev50 / 2435796066046322163

优先原因：frozen_metric_regression；结果指纹：`018c65e3b1705b13eaee9be077177e683c536c2ca12229ec56ac37db7d2eae41`。

问题：who donates a christmas tree to trafalgar square

初始：the city of Oslo, Norway

最终：The Trafalgar Square Christmas tree is a Christmas tree donated to the people of Britain by the city of Oslo , Norway each year since 1947 .

人工状态：pending，正确性与支持性均未判定。

## 16. NQ / dev50 / 5898512601992001509

优先原因：frozen_metric_regression；结果指纹：`68d51555d6d65857560e183e1aad485b0a20fad97c8d5ad48e2a721f717234b2`。

问题：who did the united states fight in the war of 1812

初始：the United Kingdom

最终：The War of 1812 ( 1812 -- 1815 ) was a conflict fought between the United States , the United Kingdom , and their respective allies .

人工状态：pending，正确性与支持性均未判定。

## 17. NQ / dev50 / 8727688733894535817

优先原因：frozen_metric_regression；结果指纹：`c0828c85a59a90f6f046f8912ce6f2b80400bc2110095985ef08e005ea353c70`。

问题：what was the purpose of the bay of pigs mission

初始：to overthrow the increasingly communist government of Fidel Castro

最终：A counter-revolutionary military group ( made up of mostly Cuban exiles who traveled to the United States after Castro 's takeover , but also of some US military personnel ) , trained and funded by the CIA , Brigade 2506 fronted the armed wing of the Democratic Revolutionary Front ( DRF ) and intended to overthrow the increasingly communist government of Fidel Castro .

人工状态：pending，正确性与支持性均未判定。

## 18. FEVER / reviewed / 156493

优先原因：frozen_metric_gain；结果指纹：`b6b6e32b49cfad6b55a9eda781d1ca1bde34644e0b3f81290aabfbc42c74ab9a`。

问题：Daggering is not a type of dance.

初始：SUPPORTS

最终：REFUTES

人工状态：pending，正确性与支持性均未判定。

## 19. FEVER / reviewed / 203615

优先原因：frozen_metric_gain；结果指纹：`820a36ef266d7747232e0ddad2984a2d23b451df2224b085c0682b20c6d3e2c3`。

问题：The 1974 crime film The Sugarland Express was co-written by Steven Spielberg.

初始：REFUTES

最终：SUPPORTS

人工状态：pending，正确性与支持性均未判定。

## 20. FEVER / reviewed / 72578

优先原因：frozen_metric_gain；结果指纹：`9ebfa78b26e1c17fd8511d7d666c6c1ef205509c5af278884fe49abeb1e54c73`。

问题：The Greek language has speakers in Albania.

初始：REFUTES

最终：SUPPORTS

人工状态：pending，正确性与支持性均未判定。

## 21. FEVER / dev50 / 203615

优先原因：frozen_metric_gain；结果指纹：`55181ab06bf23aeb3f632175451f1a4a990a2f0ca6954fb109bb3cc4bf1574d0`。

问题：The 1974 crime film The Sugarland Express was co-written by Steven Spielberg.

初始：REFUTES

最终：SUPPORTS

人工状态：pending，正确性与支持性均未判定。

## 22. NQ / reviewed / -2282501421613384225

优先原因：changed_without_frozen_correctness_gain；结果指纹：`89a22356f9f6e273d5aad34fe2a8849566ea3ca336a1bd7cc7cdc4ca4e77913d`。

问题：where is the tibia and fibula bone located

初始：on the lateral side

最终：The fibula or calf bone is a leg bone located on the lateral side of the tibia , with which it is connected above and below .

人工状态：pending，正确性与支持性均未判定。

## 23. NQ / reviewed / -2905696611778901658

优先原因：changed_without_frozen_correctness_gain；结果指纹：`7ab4cf0d320f4cd9e70359f19a53c1026174ffea7a1ca4011fa086a8ae7eb08e`。

问题：who sang oh annie i not your daddy

初始：Annie, I'm Not Your Daddy

最终：Annie , I 'm Not Your Daddy is a song written by August Darnell and first recorded by his band Kid Creole and the Coconuts .

人工状态：pending，正确性与支持性均未判定。

## 24. NQ / reviewed / 2326758035558088175

优先原因：changed_without_frozen_correctness_gain；结果指纹：`b993917b44324598c7b1984d411f44c4335495b9920052f7d844f51ed97c85a4`。

问题：what is the meaning of to have a bias

初始：prejudice in favor of or against one thing, person, or group compared with another

最终：Bias is prejudice in favour of or against one thing , person , or group compared with another , usually in a way considered to be unfair .

人工状态：pending，正确性与支持性均未判定。

## 25. NQ / reviewed / 2393389448844632496

优先原因：changed_without_frozen_correctness_gain；结果指纹：`237a814782400a05a1567c2bab882e2e6ae78f7676087a69b699eec32a66a1a8`。

问题：what is loihi and where does it originate from

初始：Lōihi Seamount

最终：Lōihi Seamount ( also known as Lōʻihi ) is an active submarine volcano about 35 km ( 22 mi ) off the southeast coast of the island of Hawaii .

人工状态：pending，正确性与支持性均未判定。

## 26. NQ / reviewed / 4534870872535404381

优先原因：changed_without_frozen_correctness_gain；结果指纹：`7b9a42a3da86b1625536c672fb8e0d5bf4647a53deaca734363cf3cda82dc447`。

问题：what went wrong on the columbia space shuttle

初始：foam insulation broke off from the Space Shuttle external tank and damaged the left wing

最终：During the launch of STS - 107 , Columbia 's 28th mission , a piece of foam insulation broke off from the Space Shuttle external tank and struck the left wing of the orbiter . A few previous shuttle launches had seen damage ranging from minor to major from foam shedding , but some engineers suspected that the damage to Columbia was more serious .

人工状态：pending，正确性与支持性均未判定。

## 27. NQ / reviewed / 6112920825267803506

优先原因：changed_without_frozen_correctness_gain；结果指纹：`b0fe9424e432dd4d1b3ceda25ff99c833adcd5cfe658c6657560eff8c2988759`。

问题：what is a serrated adenoma of the colon

初始：sessile serrated adenoma (SSA)

最终：In gastroenterology , a sessile serrated adenoma ( abbreviated SSA ) , also known as sessile serrated polyp ( abbreviated SSP ) , is a premalignant flat ( or sessile ) lesion of the colon , predominantly seen in the cecum and ascending colon .

人工状态：pending，正确性与支持性均未判定。

## 28. NQ / reviewed / 7013168592046328137

优先原因：changed_without_frozen_correctness_gain；结果指纹：`5830021a4c8985c3a786f4a7cc4efb05f67ae68cf5f6dc909f3c939b54f665b7`。

问题：who did the astros play last time they were in the world series

初始：Chicago White Sox

最终：The 2017 World Series took place between the Los Angeles Dodgers and Houston Astros .

人工状态：pending，正确性与支持性均未判定。

## 29. NQ / reviewed / 7446684003786592114

优先原因：changed_without_frozen_correctness_gain；结果指纹：`cc47f6d8620f209e59bf850fd1298740a96ebd406f68ccd9802e769f930b6974`。

问题：when does jenny come back to gossip girl

初始：October 16, 2012

最终：on October 16 , 2012 , it was reported that Momsen would be returning to Gossip Girl for the final episode of the sixth and final season , in which she appeared briefly in the wedding sequence .

人工状态：pending，正确性与支持性均未判定。

## 30. NQ / dev50 / -1209334507323554133

优先原因：changed_without_frozen_correctness_gain；结果指纹：`4d4523ac5c6e6f5306a52265e1f3970ed717a1f968b68579cd62bc39af3fe776`。

问题：what degree is a crock pot on low

初始：not specified

最终：Because the heating elements are generally located at the bottom and often also partway up the sides , most slow cookers have a minimum recommended liquid level to avoid uncontrolled heating .

人工状态：pending，正确性与支持性均未判定。

## 31. NQ / dev50 / -1680848497536972887

优先原因：changed_without_frozen_correctness_gain；结果指纹：`135391a819505cb617f23aa99f7670bcf155aa087a640216672c309386d89406`。

问题：what is property of a system in thermodynamics

初始：any property that describes a state of a physical system

最终：Within thermodynamics , a physical property is any property that is measurable , and whose value describes a state of a physical system .

人工状态：pending，正确性与支持性均未判定。

## 32. NQ / dev50 / -1814654291537702625

优先原因：changed_without_frozen_correctness_gain；结果指纹：`9ff30adfe7c529476c1a03b5389341c767a95c1397473ce48514ede96b42d5c0`。

问题：what is the basic principle of christian morality

初始：love

最终：The New Testament generally asserts that all morality flows from the Great Commandment , to love God with all one 's heart , mind , strength , and soul , and to love one 's neighbour as oneself .

人工状态：pending，正确性与支持性均未判定。

## 33. NQ / dev50 / -2282501421613384225

优先原因：changed_without_frozen_correctness_gain；结果指纹：`93117f02576eb6e2348325756e8e028b0b036451ea3d7d3387e45b9cec6070d1`。

问题：where is the tibia and fibula bone located

初始：on the lateral side

最终：The fibula or calf bone is a leg bone located on the lateral side of the tibia , with which it is connected above and below .

人工状态：pending，正确性与支持性均未判定。

## 34. NQ / dev50 / -2743701347497238377

优先原因：changed_without_frozen_correctness_gain；结果指纹：`898d964112f2229dcdc5445f470088c8b904eaf87a22ef2414c2b6505f9942c3`。

问题：how oxygenated blood returns to the heart from the lungs

初始：pulmonary veins

最终：The oxygenated blood then leaves the lungs through pulmonary veins , which return it to the left heart , completing the pulmonary cycle .

人工状态：pending，正确性与支持性均未判定。

## 35. NQ / dev50 / -2905696611778901658

优先原因：changed_without_frozen_correctness_gain；结果指纹：`7a236bf83bd5dbcad8b7fe6a0cd286e527defb51a2d5bad2aa317a74e6efb9d1`。

问题：who sang oh annie i not your daddy

初始：Annie, I'm Not Your Daddy

最终：Annie , I 'm Not Your Daddy is a song written by August Darnell and first recorded by his band Kid Creole and the Coconuts .

人工状态：pending，正确性与支持性均未判定。

## 36. NQ / dev50 / -3089533754017582743

优先原因：changed_without_frozen_correctness_gain；结果指纹：`d3cb3d2822d82372b820e8aa98963d38d8a09b90012aeb3921040e5e311ac596`。

问题：when was the last time unc did not make the ncaa tournament

初始：2001

最终：The state of North Carolina was threatened with a 2018 - 2022 championship venue boycott by the NCAA , due to the HB2 law passed in 2016 .

人工状态：pending，正确性与支持性均未判定。

## 37. NQ / dev50 / -5973003791441271262

优先原因：changed_without_frozen_correctness_gain；结果指纹：`4059b317bd0b75c897c27f17c135c966a0329a045ff017350bb6280b16c69f9b`。

问题：where does the last name andersen originate from

初始：Danish-Norwegian

最终：Andersen ( Danish pronunciation : ( ˈɑnɐsn̩ ) ) is a Danish - Norwegian patronymic surname meaning `` son of Anders '' ( itself derived from the Greek name `` Ανδρέας / Andreas '' , cf . English Andrew ) .

人工状态：pending，正确性与支持性均未判定。

## 38. NQ / dev50 / 2393389448844632496

优先原因：changed_without_frozen_correctness_gain；结果指纹：`1f7247f570ef8a83c119d9a4c445f0cf7f30e0226a1834fd1ddf40fda345b1c2`。

问题：what is loihi and where does it originate from

初始：Lōihi Seamount

最终：Lōihi Seamount ( also known as Lōʻihi ) is an active submarine volcano about 35 km ( 22 mi ) off the southeast coast of the island of Hawaii .

人工状态：pending，正确性与支持性均未判定。

## 39. NQ / dev50 / 6781489363263681689

优先原因：changed_without_frozen_correctness_gain；结果指纹：`f63ccc82794a7a6b8219a2f135de8de5d1a2c737180647b43faf0caef6966a07`。

问题：where does the last name rudolph come from

初始：Germanic name

最终：Rudolph or Rudolf ( French : Rodolphe , Italian , Portuguese and Spanish : Rodolfo ) or Rodolphe is a male first name , and , less commonly , a surname . It is a Germanic name deriving from two stems : Rod or Hrōð , meaning `` fame '' , and olf meaning `` wolf '' ( see also Hroðulf ; cf . Adolf ) .

人工状态：pending，正确性与支持性均未判定。

## 40. NQ / dev50 / 7632941217874741929

优先原因：changed_without_frozen_correctness_gain；结果指纹：`ae59ee0305a75755654f660db9637bf809c6fab89003bab945a2dcf685f75a2c`。

问题：what is george harrison song wah wah about

初始：frustration with the atmosphere in the group

最终：The lyrics reflect his frustration with the atmosphere in the group at that time -- namely , Paul McCartney 's over-assertiveness and criticism of his guitar playing , John Lennon 's lack of engagement with the project and dismissal of Harrison as a songwriter , and Yoko Ono 's constant involvement in the band 's activities .

人工状态：pending，正确性与支持性均未判定。

## 41. NQ / dev50 / 8003762008471695244

优先原因：changed_without_frozen_correctness_gain；结果指纹：`3b60f197bd75588cdea75702fb04d870872ad92eba5ef2996194bea32dd4149e`。

问题：the christian church used which holiday to take attention away from samhain

初始：Christmas

最终：It was a custom of the Pagans to celebrate on the same 25 December the birthday of the Sun , at which they kindled lights in token of festivity . In these solemnities and revelries the Christians also took part . Accordingly when the doctors of the Church perceived that the Christians had a leaning to this festival , they took counsel and resolved that the true Nativity should be solemnised on that day .

人工状态：pending，正确性与支持性均未判定。

## 42. NQ / dev50 / 970985133701043298

优先原因：changed_without_frozen_correctness_gain；结果指纹：`b35ada7901232f1c565b267f02615ce7e52450f52e4a62af9544a19714ae36b7`。

问题：a legislative act passed by congress is an example of

初始：statute

最终：The act was passed by large majorities in both houses of Congress .

人工状态：pending，正确性与支持性均未判定。

