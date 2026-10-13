# E v6r1：128 条结果辅助审核

2026-10-14，Codex 代理辅助判断，全部待用户确认。审核使用冻结问题、初始/最终答案、参考标签/答案及 C1 Top-5，参考答案可见，属于非盲开发审核。

按主题定位和审阅相关证据句、必要的完整段落与模型实际引用窗口；并非独立人工全文盲审。没有外部事实检索。引用片段均已在对应存档中逐字定位。

“正确”判断是否回答问题的目标、关系、时间和完整性；允许同义短答、较长答案。“支持”判断 C1 是否足以支持该回答；合理拒答可以有支持但不算回答成功。事实短语真实但不回答本题，不算问题所需的支持。

Top-5 支持与模型实际引用分开。缺少证据不等于反驳。FEVER 某些正确性仅依据冻结标签，已标 reference_dependent_correctness；这些不是独立事实确认。时间依赖问题采用冻结语料时点。

128 条结果含 123 个唯一样本。5 条重复记录的审核材料完全相同，复用判断并保留各自结果 SHA-256。人工 pair_review 未改，agent_review 独立存放。待定项以 null 保存。

| 批次 | 条数 | 建议语义修复 | 建议语义误修 | 正确性待定 |
|---|---:|---:|---:|---:|
| fever/dev50/fever_nli | 50 | 1 | 2 | 0 |
| nq/reviewed/extractive_reanswer | 14 | 6 | 0 | 0 |
| nq/dev50/extractive_reanswer | 50 | 3 | 1 | 4 |
| fever/reviewed/fever_nli | 14 | 3 | 0 | 0 |

以上是代理建议数；跨批次有重叠，不直接相加宣传效果。对“正确性待定”没有强行计入修复/误修。

## 1. FEVER / dev50 / 140820

问题：Men in Black II is a film from America.

初始答案：SUPPORTS

最终答案：REFUTES

建议：初始正确 **是** → 最终正确 **否**；Top-5 支持 **否 → 否**。

实际输出引用：不支持。置信度：high。

初始 SUPPORTS 符合冻结标签；Top-5 的 Men in Black 是第一部，未直接证明第二部国别。最终引用比利时电影 Black，实体错误，不能推出 Men in Black II 非美国电影。判为答案误修，初始答案也缺少完整 C1 支持。

证据定位：R1 `Men_in_Black_-LRB-film-RRB-`：Men in Black is a 1997 American；R2 `Black_-LRB-2015_Belgian_film-RRB-`：Black is a 2015 Belgian

冻结参考（仅作对照）：SUPPORTS

结果 SHA-256：`5017ca4bd2f346b8d66663c832be083adff27c902ac1b212313375382879e15b`；人工状态：pending。

## 2. FEVER / dev50 / 33982

问题：Henry Cavill starred in two Superman films.

初始答案：SUPPORTS

最终答案：REFUTES

建议：初始正确 **是** → 最终正确 **否**；Top-5 支持 **否 → 否**。

实际输出引用：不支持。置信度：high。

初始 SUPPORTS 符合冻结标签；C1 明确 Cavill 主演 Man of Steel，但没有完整证明他主演两部。最终引用 Henry's Crime，与 Cavill/Superman 无关，改成 REFUTES 是误修。

证据定位：R1 `Man_of_Steel_-LRB-film-RRB-`：stars Henry Cavill；R5 `Henry's_Crime`：Henry 's Crime is a 2010

冻结参考（仅作对照）：SUPPORTS

结果 SHA-256：`a465787cdacf4a3c272cc098daf71dc2c96b5e0d6e4cfe24625e27b019935f4c`；人工状态：pending。

## 3. NQ / reviewed / 5259818422556035007

问题：who sings good night on the beatles white album

初始答案：Ringo Starr

最终答案：It is sung by Ringo Starr , the only Beatle to appear on the track .

建议：初始正确 **是** → 最终正确 **是**；Top-5 支持 **是 → 是**。

实际输出引用：结合原段指代后支持。置信度：high。

前后都明确演唱者 Ringo Starr；最终 It 的指代可由同一 R1 的 Good Night/White Album 语境确定。EM 下降只是扩成长句，不是语义误修。

证据定位：R1 `nq_p_013331`：It is sung by Ringo Starr , the only Beatle to appear on the track .

冻结参考（仅作对照）：Ringo Starr

结果 SHA-256：`11d506c9044436059e2bb23850ab27f9b8318392c6f90b48169a71b1344349d3`；人工状态：pending。

## 4. NQ / reviewed / 970104939222709777

问题：who is credited with creating the gothic art movement

初始答案：Abbot Suger

最终答案：Gothic art emerged in Île - de-France , France , in the early 12th century at the Abbey Church of St Denis built by Abbot Suger .

建议：初始正确 **是** → 最终正确 **是**；Top-5 支持 **是 → 是**。

实际输出引用：支持。置信度：medium。

初始 Abbot Suger 与冻结参考及 Gothic art 起源于其建造的 St Denis 的材料一致。最终保留人物并补充背景，属于有支持的扩写；将 credited with creating 理解为材料中的发源/推动者，沿用先前人工确认。

证据定位：R1 `nq_p_072993`：at the Abbey Church of St Denis built by Abbot Suger

冻结参考（仅作对照）：Abbot Suger

结果 SHA-256：`279dd6c9369dc81eabb71fa882d757bf3c7f25f0d148fd479710da21d6577d35`；人工状态：pending。

## 5. NQ / dev50 / -1450346568093066821

问题：where was because i could not stop for death published

初始答案：Poems: Series 1

最终答案：Because I could not stop for Death '' is a lyrical poem by Emily Dickinson first published posthumously in Poems : Series 1 in 1890 .

建议：初始正确 **是** → 最终正确 **是**；Top-5 支持 **是 → 是**。

实际输出引用：支持。置信度：high。

前后都回答发表于 Poems: Series 1；最终补充作者及 1890 年，均在 R1。仅因整句而 EM 下降。

证据定位：R1 `nq_p_018799`：first published posthumously in Poems : Series 1 in 1890

冻结参考（仅作对照）：Poems: Series 1

结果 SHA-256：`b47707d53914b86ac444f1dd69fa52325c4aa472aebadcde07ebee0ec4fb7531`；人工状态：pending。

## 6. NQ / dev50 / -2680244887428298888

问题：how many lines of symmetry are there in a equilateral triangle

初始答案：3

最终答案：An equilateral triangle is the most symmetrical triangle , having 3 lines of reflection and rotational symmetry of order 3 about its center .

建议：初始正确 **是** → 最终正确 **是**；Top-5 支持 **是 → 是**。

实际输出引用：支持。置信度：high。

前后都给出 3 条对称轴；R1 的 3 lines of reflection 直接支持。整句另提旋转对称也来自原文，不构成误修。

证据定位：R1 `nq_p_062750`：having 3 lines of reflection

冻结参考（仅作对照）：3

结果 SHA-256：`998669c4a1fe4c33eb21c212825710e263ea57ad3cd6d3b93939f3364869e26b`；人工状态：pending。

## 7. NQ / dev50 / -3064086919220674644

问题：when did the great plague of london start and end

初始答案：1665 to 1666

最终答案：The Great Plague , lasting from 1665 to 1666 , was the last major epidemic of the bubonic plague to occur in England .

建议：初始正确 **是** → 最终正确 **是**；Top-5 支持 **是 → 是**。

实际输出引用：支持。置信度：high。

前后都给出 1665–1666；R1 直接记录 Great Plague 的时间区间，长句导致 EM 下降。

证据定位：R1 `nq_p_079663`：The Great Plague , lasting from 1665 to 1666

冻结参考（仅作对照）：1665 to 1666

结果 SHA-256：`816314b440f42e262c4e22384313287daa83304131fec48ad61228a27ca5f63c`；人工状态：pending。

## 8. NQ / dev50 / -3339006116507262453

问题：who sang i ran all the way home

初始答案：The Impalas

最终答案：The Impalas were an American doo - wop group in the late 1950s , best known for their hit , `` Sorry ( I Ran All the Way Home ) '' .

建议：初始正确 **是** → 最终正确 **是**；Top-5 支持 **是 → 是**。

实际输出引用：支持。置信度：high。

R2 明确 The Impalas 的歌曲 Sorry (I Ran All the Way Home)，前后演唱者一致，最终为正确扩写。

证据定位：R2 `nq_p_056375`：The Impalas were an American doo - wop group

冻结参考（仅作对照）：The Impalas

结果 SHA-256：`1464ebe8e1bab6273dd22b9b44b697e6e3e7f7e68f4b70646562b32e6d3fa566`；人工状态：pending。

## 9. NQ / dev50 / -5748368670037082784

问题：what is the definition of the name thomas

初始答案：twin

最终答案：Thomas was a popular medieval given throughout Europe , coming from the biblical name , Thomas being one of Jesus ' disciples . It is ultimately derived from the Aramaic personal name meaning `` twin . ''

建议：初始正确 **是** → 最终正确 **是**；Top-5 支持 **是 → 是**。

实际输出引用：结合原段指代后支持。置信度：high。

Thomas 的词源含义为 twin；最终补上词源背景并保留 twin，R1 直接支持。

证据定位：R1 `nq_p_030047`：It is ultimately derived from the Aramaic personal name meaning

冻结参考（仅作对照）：twin

结果 SHA-256：`35422eac193e4faf37b37021d60baa483c1b895e0064396ea315a10a54641f48`；人工状态：pending。

## 10. NQ / dev50 / -6089464774983144695

问题：who plays grace in the secret life of the american teenager

初始答案：Megan Park

最终答案：Megan Park ( born July 24 , 1986 ) is a Canadian actress and singer . She is known for her portrayal of Grace Bowman in the television series The Secret Life of the American Teenager .

建议：初始正确 **是** → 最终正确 **是**；Top-5 支持 **是 → 是**。

实际输出引用：支持。置信度：high。

R1 明确 Megan Park 扮演 Grace Bowman，前后答案正确；最终的人物背景也来自同段。

证据定位：R1 `nq_p_014498`：She is known for her portrayal of Grace Bowman

冻结参考（仅作对照）：Megan Park

结果 SHA-256：`1f7bcb98f0303333d0897d235bf635c6c01ab6349e325cf293e5a69dd9271890`；人工状态：pending。

## 11. NQ / dev50 / -8209571805051987276

问题：what is the genus of a bald eagle

初始答案：Haliaeetus

最终答案：The bald eagle placed in the genus Haliaeetus

建议：初始正确 **是** → 最终正确 **是**；Top-5 支持 **是 → 是**。

实际输出引用：支持。置信度：high。

R1 明确 bald eagle 的属为 Haliaeetus；最终只是将名称放回原文短句，没有语义退步。

证据定位：R1 `nq_p_030261`：The bald eagle placed in the genus Haliaeetus

冻结参考（仅作对照）：Haliaeetus

结果 SHA-256：`b62d234ca60df73d791546348485b4ae9c0f419358926ad29c5d9c8c6cc0d53d`；人工状态：pending。

## 12. NQ / dev50 / -8631190913794477985

问题：who scored the most points in a single game in the nba

初始答案：Wilt Chamberlain

最终答案：Chamberlain holds the single - game scoring record , having scored 100 in game in 1962 .

建议：初始正确 **是** → 最终正确 **是**；Top-5 支持 **是 → 是**。

实际输出引用：结合原段指代后支持。置信度：high。

R1 先点名 Wilt Chamberlain，再说明 Chamberlain 单场 100 分纪录；前后都回答同一人。引用姓氏需同段前文消歧，按冻结语料判断。

证据定位：R1 `nq_p_045105`：Wilt Chamberlain ( 32 times )；R1 `nq_p_045105`：Chamberlain holds the single - game scoring record

冻结参考（仅作对照）：Wilt Chamberlain

结果 SHA-256：`a075bf0b926cabce1459515094c24c7107b413c1d4fe9cf54d0c06de638809e2`；人工状态：pending。

## 13. NQ / dev50 / -8825841420882619284

问题：where's the tv show the crossing filmed

初始答案：British Columbia, Canada

最终答案：Set in the fictional town of Port Canaan , Oregon and in Seattle , the series was filmed in coastal areas of British Columbia and in the city of Vancouver , British Columbia in 2017 .

建议：初始正确 **是** → 最终正确 **是**；Top-5 支持 **是 → 是**。

实际输出引用：结合原段指代后支持。置信度：high。

R1 指定 The Crossing 在 British Columbia, Canada 拍摄；R2 扩充沿海和 Vancouver。最终区分 fictional setting 与 filmed in，没有把 Oregon 当拍摄地。

证据定位：R1 `nq_p_013993`：The series is filmed in British Columbia , Canada .；R2 `nq_p_013995`：the series was filmed in coastal areas of British Columbia

冻结参考（仅作对照）：British Columbia, Canada

结果 SHA-256：`667c261b602fc98769c83ff9603a5a3f6abf7c1862c25a2aacbd4f88436e5194`；人工状态：pending。

## 14. NQ / dev50 / 1117400623925969758

问题：who won the ncaa women's championship last year 2017

初始答案：South Carolina

最终答案：South Carolina defeated Mississippi State to win the championship .

建议：初始正确 **是** → 最终正确 **是**；Top-5 支持 **是 → 是**。

实际输出引用：结合原段指代后支持。置信度：high。

R1 的 2017 NCAA 女篮赛事语境明确 South Carolina 获胜；前后正确，最终加对手不改变答案。

证据定位：R1 `nq_p_046560`：South Carolina defeated Mississippi State to win the championship .

冻结参考（仅作对照）：South Carolina

结果 SHA-256：`d9c89e96fa33470f213618918e7ce3a922e855504f3601f5fed8a80af7d4923e`；人工状态：pending。

## 15. NQ / dev50 / 2435796066046322163

问题：who donates a christmas tree to trafalgar square

初始答案：the city of Oslo, Norway

最终答案：The Trafalgar Square Christmas tree is a Christmas tree donated to the people of Britain by the city of Oslo , Norway each year since 1947 .

建议：初始正确 **是** → 最终正确 **是**；Top-5 支持 **是 → 是**。

实际输出引用：支持。置信度：high。

前后都回答奥斯陆市赠树，R1 包含对象、地点和每年赠送的关系。

证据定位：R1 `nq_p_010450`：by the city of Oslo , Norway each year since 1947

冻结参考（仅作对照）：the city of Oslo, Norway

结果 SHA-256：`018c65e3b1705b13eaee9be077177e683c536c2ca12229ec56ac37db7d2eae41`；人工状态：pending。

## 16. NQ / dev50 / 5898512601992001509

问题：who did the united states fight in the war of 1812

初始答案：the United Kingdom

最终答案：The War of 1812 ( 1812 -- 1815 ) was a conflict fought between the United States , the United Kingdom , and their respective allies .

建议：初始正确 **是** → 最终正确 **是**；Top-5 支持 **是 → 是**。

实际输出引用：支持。置信度：high。

R1 明确战争双方为美国、英国及各自盟友；最终扩成原句仍正确回答英国。

证据定位：R1 `nq_p_015685`：between the United States , the United Kingdom , and their respective allies

冻结参考（仅作对照）：United Kingdom

结果 SHA-256：`68d51555d6d65857560e183e1aad485b0a20fad97c8d5ad48e2a721f717234b2`；人工状态：pending。

## 17. NQ / dev50 / 8727688733894535817

问题：what was the purpose of the bay of pigs mission

初始答案：to overthrow the increasingly communist government of Fidel Castro

最终答案：A counter-revolutionary military group ( made up of mostly Cuban exiles who traveled to the United States after Castro 's takeover , but also of some US military personnel ) , trained and funded by the CIA , Brigade 2506 fronted the armed wing of the Democratic Revolutionary Front ( DRF ) and intended to overthrow the increasingly communist government of Fidel Castro .

建议：初始正确 **是** → 最终正确 **是**；Top-5 支持 **是 → 是**。

实际输出引用：结合原段指代后支持。置信度：high。

前后都回答推翻 Castro 政府；R1 把该目的与 Bay of Pigs/Brigade 2506 相连。最终冗长但内容有证据。

证据定位：R1 `nq_p_024937`：intended to overthrow the increasingly communist government of Fidel Castro

冻结参考（仅作对照）：to overthrow the increasingly communist government of Fidel Castro

结果 SHA-256：`c0828c85a59a90f6f046f8912ce6f2b80400bc2110095985ef08e005ea353c70`；人工状态：pending。

## 18. FEVER / reviewed / 156493

问题：Daggering is not a type of dance.

初始答案：SUPPORTS

最终答案：REFUTES

建议：初始正确 **否** → 最终正确 **是**；Top-5 支持 **否 → 是**。

实际输出引用：支持。置信度：high。

Daggering 明确是舞蹈；not a traditional dance 只否定传统舞蹈身份，不否定它是舞蹈。由 SUPPORTS 改为 REFUTES 正确，引用直接支持反驳。

证据定位：R1 `Daggering`：Daggering is a form of dance originating from Jamaica .

冻结参考（仅作对照）：REFUTES

结果 SHA-256：`b6b6e32b49cfad6b55a9eda781d1ca1bde34644e0b3f81290aabfbc42c74ab9a`；人工状态：pending。

## 19. FEVER / reviewed / 203615

问题：The 1974 crime film The Sugarland Express was co-written by Steven Spielberg.

初始答案：REFUTES

最终答案：SUPPORTS

建议：初始正确 **否** → 最终正确 **是**；Top-5 支持 **否 → 是**。

实际输出引用：支持。置信度：high。

R1 同一句明确 1974、crime drama、co-written by Steven Spielberg；初始反驳错误，最终支持正确且引用完整。两批实验重复出现，不算两次独立修复。

证据定位：R1 `The_Sugarland_Express`：The Sugarland Express is a 1974 American crime drama film co-written and directed by Steven Spielberg

冻结参考（仅作对照）：SUPPORTS

结果 SHA-256：`820a36ef266d7747232e0ddad2984a2d23b451df2224b085c0682b20c6d3e2c3`；人工状态：pending。

## 20. FEVER / reviewed / 72578

问题：The Greek language has speakers in Albania.

初始答案：REFUTES

最终答案：SUPPORTS

建议：初始正确 **否** → 最终正确 **是**；Top-5 支持 **否 → 是**。

实际输出引用：支持。置信度：high。

Languages of Albania 明确希腊语在阿尔巴尼亚南部广泛使用，足以支持当地有希腊语使用者。最终 SUPPORTS 是有证据的修复。

证据定位：R1 `Languages_of_Albania`：Greek , the language of the Greek minority of the south , is also very widespread in that region .

冻结参考（仅作对照）：SUPPORTS

结果 SHA-256：`9ebfa78b26e1c17fd8511d7d666c6c1ef205509c5af278884fe49abeb1e54c73`；人工状态：pending。

## 21. FEVER / dev50 / 203615

问题：The 1974 crime film The Sugarland Express was co-written by Steven Spielberg.

初始答案：REFUTES

最终答案：SUPPORTS

建议：初始正确 **否** → 最终正确 **是**；Top-5 支持 **否 → 是**。

实际输出引用：支持。置信度：high。

R1 同一句明确 1974、crime drama、co-written by Steven Spielberg；初始反驳错误，最终支持正确且引用完整。两批实验重复出现，不算两次独立修复。

证据定位：R1 `The_Sugarland_Express`：The Sugarland Express is a 1974 American crime drama film co-written and directed by Steven Spielberg

冻结参考（仅作对照）：SUPPORTS

结果 SHA-256：`55181ab06bf23aeb3f632175451f1a4a990a2f0ca6954fb109bb3cc4bf1574d0`；人工状态：pending。

本条与第 19 条审核材料相同；独立保留本批结果指纹。

## 22. NQ / reviewed / -2282501421613384225

问题：where is the tibia and fibula bone located

初始答案：on the lateral side

最终答案：The fibula or calf bone is a leg bone located on the lateral side of the tibia , with which it is connected above and below .

建议：初始正确 **否** → 最终正确 **是**；Top-5 支持 **否 → 是**。

实际输出引用：支持。置信度：medium。

原答案只给相对位置 lateral side，未回答身体部位；最终明确 leg bone 并说明与 tibia 上下相连，按参考要求的 leg 粒度，建议判为已补足部位的修复。若要求明确两骨均位于小腿，应进一步精简改写，留人工确认。

证据定位：R1 `nq_p_006327`：The fibula or calf bone is a leg bone located on the lateral side of the tibia

冻结参考（仅作对照）：leg

结果 SHA-256：`89a22356f9f6e273d5aad34fe2a8849566ea3ca336a1bd7cc7cdc4ca4e77913d`；人工状态：pending。

## 23. NQ / reviewed / -2905696611778901658

问题：who sang oh annie i not your daddy

初始答案：Annie, I'm Not Your Daddy

最终答案：Annie , I 'm Not Your Daddy is a song written by August Darnell and first recorded by his band Kid Creole and the Coconuts .

建议：初始正确 **否** → 最终正确 **是**；Top-5 支持 **否 → 是**。

实际输出引用：支持。置信度：high。

原答案重复歌名，未给演唱者；最终明确 Kid Creole and the Coconuts 首次录制，回答了谁唱，引用完整。EM 未识别这一语义修复。

证据定位：R1 `nq_p_001994`：first recorded by his band Kid Creole and the Coconuts

冻结参考（仅作对照）：Kid Creole and the Coconuts

结果 SHA-256：`7ab4cf0d320f4cd9e70359f19a53c1026174ffea7a1ca4011fa086a8ae7eb08e`；人工状态：pending。

## 24. NQ / reviewed / 2326758035558088175

问题：what is the meaning of to have a bias

初始答案：prejudice in favor of or against one thing, person, or group compared with another

最终答案：Bias is prejudice in favour of or against one thing , person , or group compared with another , usually in a way considered to be unfair .

建议：初始正确 **是** → 最终正确 **是**；Top-5 支持 **是 → 是**。

实际输出引用：支持。置信度：high。

原答案已正确解释 bias 为偏向或反对某对象的偏见，此前人工确认；最终补 usually unfair，内容更完整，但不算错误修复。

证据定位：R1 `nq_p_004698`：Bias is prejudice in favour of or against one thing

冻结参考（仅作对照）：prejudice in favour of or against one thing, person, or group compared with another, usually in a way considered to be unfair

结果 SHA-256：`b993917b44324598c7b1984d411f44c4335495b9920052f7d844f51ed97c85a4`；人工状态：pending。

## 25. NQ / reviewed / 2393389448844632496

问题：what is loihi and where does it originate from

初始答案：Lōihi Seamount

最终答案：Lōihi Seamount ( also known as Lōʻihi ) is an active submarine volcano about 35 km ( 22 mi ) off the southeast coast of the island of Hawaii .

建议：初始正确 **否** → 最终正确 **是**；Top-5 支持 **否 → 是**。

实际输出引用：支持。置信度：high。

原答案仅重复 Lōihi Seamount 名称；最终补足它是海底活火山及位于夏威夷岛东南近海，满足冻结问题/参考的定义与位置要求。

证据定位：R1 `nq_p_099602`：is an active submarine volcano about 35 km ( 22 mi ) off the southeast coast of the island of Hawaii

冻结参考（仅作对照）：an active submarine volcano about 35 km (22 mi) off the southeast coast of the island of Hawaii

结果 SHA-256：`237a814782400a05a1567c2bab882e2e6ae78f7676087a69b699eec32a66a1a8`；人工状态：pending。

## 26. NQ / reviewed / 4534870872535404381

问题：what went wrong on the columbia space shuttle

初始答案：foam insulation broke off from the Space Shuttle external tank and damaged the left wing

最终答案：During the launch of STS - 107 , Columbia 's 28th mission , a piece of foam insulation broke off from the Space Shuttle external tank and struck the left wing of the orbiter . A few previous shuttle launches had seen damage ranging from minor to major from foam shedding , but some engineers suspected that the damage to Columbia was more serious .

建议：初始正确 **是** → 最终正确 **是**；Top-5 支持 **是 → 是**。

实际输出引用：支持。置信度：high。

原答案已给出外部燃料箱泡沫脱落并损伤左翼这一事故原因，此前确认正确；最终是同一因果内容的有据扩写，不是新修复。

证据定位：R1 `nq_p_039123`：a piece of foam insulation broke off from the Space Shuttle external tank and struck the left wing

冻结参考（仅作对照）：a piece of foam insulation broke off from the Space Shuttle external tank and struck the left wing of the orbiter

结果 SHA-256：`7b9a42a3da86b1625536c672fb8e0d5bf4647a53deaca734363cf3cda82dc447`；人工状态：pending。

## 27. NQ / reviewed / 6112920825267803506

问题：what is a serrated adenoma of the colon

初始答案：sessile serrated adenoma (SSA)

最终答案：In gastroenterology , a sessile serrated adenoma ( abbreviated SSA ) , also known as sessile serrated polyp ( abbreviated SSP ) , is a premalignant flat ( or sessile ) lesion of the colon , predominantly seen in the cecum and ascending colon .

建议：初始正确 **否** → 最终正确 **是**；Top-5 支持 **否 → 是**。

实际输出引用：支持。置信度：high。

原答案只重复病名/缩写，未作定义；最终明确结肠的癌前扁平病变并给位置，按冻结材料回答了 what is。

证据定位：R1 `nq_p_016576`：is a premalignant flat ( or sessile ) lesion of the colon

冻结参考（仅作对照）：a premalignant flat (or sessile) lesion of the colon

结果 SHA-256：`b0fe9424e432dd4d1b3ceda25ff99c833adcd5cfe658c6657560eff8c2988759`；人工状态：pending。

## 28. NQ / reviewed / 7013168592046328137

问题：who did the astros play last time they were in the world series

初始答案：Chicago White Sox

最终答案：The 2017 World Series took place between the Los Angeles Dodgers and Houston Astros .

建议：初始正确 **否** → 最终正确 **是**；Top-5 支持 **否 → 是**。

实际输出引用：支持。置信度：high。

R1 区分 2005 对手 White Sox 和 2017 对手 Dodgers；按冻结语料中的 last time，原答案选旧事件，最终改为 2017 Dodgers，修复正确。

证据定位：R1 `nq_p_018346`：in 2005 , against the Chicago White Sox；R2 `nq_p_011513`：The 2017 World Series took place between the Los Angeles Dodgers and Houston Astros .

冻结参考（仅作对照）：Los Angeles Dodgers

结果 SHA-256：`5830021a4c8985c3a786f4a7cc4efb05f67ae68cf5f6dc909f3c939b54f665b7`；人工状态：pending。

## 29. NQ / reviewed / 7446684003786592114

问题：when does jenny come back to gossip girl

初始答案：October 16, 2012

最终答案：on October 16 , 2012 , it was reported that Momsen would be returning to Gossip Girl for the final episode of the sixth and final season , in which she appeared briefly in the wedding sequence .

建议：初始正确 **否** → 最终正确 **是**；Top-5 支持 **否 → 是**。

实际输出引用：结合原段指代后支持。置信度：high。

原答案把回归报道日期当回归时间。最终保留 reported 的关系，并明确第六季/全剧最终集回归；R1/R3 支持，建议判为语义修复。

证据定位：R1 `nq_p_035769`：it was reported that Momsen would be returning to Gossip Girl for the final episode；R3 `nq_p_050741`：In the final episode of the series

冻结参考（仅作对照）：the final episode of the series

结果 SHA-256：`cc47f6d8620f209e59bf850fd1298740a96ebd406f68ccd9802e769f930b6974`；人工状态：pending。

## 30. NQ / dev50 / -1209334507323554133

问题：what degree is a crock pot on low

初始答案：not specified

最终答案：Because the heating elements are generally located at the bottom and often also partway up the sides , most slow cookers have a minimum recommended liquid level to avoid uncontrolled heating .

建议：初始正确 **否** → 最终正确 **否**；Top-5 支持 **是 → 否**。

实际输出引用：不支持。置信度：high。

C1 没给 low 档确定温度，原 not specified 属合理拒答但未回答所求数值。最终转成最低液位/加热位置说明，仍未回答温度。冻结参考 71–74°C 在 R5 实际对应 warming，疑似档位错配，不能用它判 low 的事实真值。

证据定位：R3 `nq_p_063087`：At a lower setting , it may just simmer；R5 `nq_p_063088`：switch from cooking to warming；R1 `nq_p_063086`：minimum recommended liquid level

冻结参考（仅作对照）：71–74 °C (160–165 °F)

结果 SHA-256：`4d4523ac5c6e6f5306a52265e1f3970ed717a1f968b68579cd62bc39af3fe776`；人工状态：pending。

## 31. NQ / dev50 / -1680848497536972887

问题：what is property of a system in thermodynamics

初始答案：any property that describes a state of a physical system

最终答案：Within thermodynamics , a physical property is any property that is measurable , and whose value describes a state of a physical system .

建议：初始正确 **待定** → 最终正确 **是**；Top-5 支持 **是 → 是**。

实际输出引用：支持。置信度：medium。

原答案给出描述系统状态这一核心含义，但省略 measurable 和 value，是否足够完整有边界；最终完整保留这两个条件并直接来自 R1。最终正确，暂不把它计为确定修复。

证据定位：R1 `nq_p_073593`：any property that is measurable , and whose value describes a state of a physical system

冻结参考（仅作对照）：any property that is measurable, and whose value describes a state of a physical system

结果 SHA-256：`135391a819505cb617f23aa99f7670bcf155aa087a640216672c309386d89406`；人工状态：pending。

## 32. NQ / dev50 / -1814654291537702625

问题：what is the basic principle of christian morality

初始答案：love

最终答案：The New Testament generally asserts that all morality flows from the Great Commandment , to love God with all one 's heart , mind , strength , and soul , and to love one 's neighbour as oneself .

建议：初始正确 **待定** → 最终正确 **是**；Top-5 支持 **是 → 是**。

实际输出引用：支持。置信度：medium。

love 与 R2 方向一致，但是否足以回答基本原则需确认；最终明确爱神与爱邻如己，完整且有直接支持。暂不将扩写计为确定修复。

证据定位：R2 `nq_p_030136`：all morality flows from the Great Commandment

冻结参考（仅作对照）：all morality flows from the Great Commandment, to love God with all one's heart, mind, strength, and soul, and to love one's neighbour as oneself

结果 SHA-256：`9ff30adfe7c529476c1a03b5389341c767a95c1397473ce48514ede96b42d5c0`；人工状态：pending。

## 33. NQ / dev50 / -2282501421613384225

问题：where is the tibia and fibula bone located

初始答案：on the lateral side

最终答案：The fibula or calf bone is a leg bone located on the lateral side of the tibia , with which it is connected above and below .

建议：初始正确 **否** → 最终正确 **是**；Top-5 支持 **否 → 是**。

实际输出引用：支持。置信度：medium。

原答案只给相对位置 lateral side，未回答身体部位；最终明确 leg bone 并说明与 tibia 上下相连，按参考要求的 leg 粒度，建议判为已补足部位的修复。若要求明确两骨均位于小腿，应进一步精简改写，留人工确认。

证据定位：R1 `nq_p_006327`：The fibula or calf bone is a leg bone located on the lateral side of the tibia

冻结参考（仅作对照）：leg

结果 SHA-256：`93117f02576eb6e2348325756e8e028b0b036451ea3d7d3387e45b9cec6070d1`；人工状态：pending。

本条与第 22 条审核材料相同；独立保留本批结果指纹。

## 34. NQ / dev50 / -2743701347497238377

问题：how oxygenated blood returns to the heart from the lungs

初始答案：pulmonary veins

最终答案：The oxygenated blood then leaves the lungs through pulmonary veins , which return it to the left heart , completing the pulmonary cycle .

建议：初始正确 **是** → 最终正确 **是**；Top-5 支持 **是 → 是**。

实际输出引用：支持。置信度：high。

题问含氧血如何从肺回心，R1/R2 直接支持经肺静脉；原 pulmonary veins 已正确。参考 pulmonary circulation 粒度更宽，不能因此判原答案错。最终只是完整表述。

证据定位：R1 `nq_p_090777`：through pulmonary veins , which return it to the left heart

冻结参考（仅作对照）：pulmonary circulation

结果 SHA-256：`898d964112f2229dcdc5445f470088c8b904eaf87a22ef2414c2b6505f9942c3`；人工状态：pending。

## 35. NQ / dev50 / -2905696611778901658

问题：who sang oh annie i not your daddy

初始答案：Annie, I'm Not Your Daddy

最终答案：Annie , I 'm Not Your Daddy is a song written by August Darnell and first recorded by his band Kid Creole and the Coconuts .

建议：初始正确 **否** → 最终正确 **是**；Top-5 支持 **否 → 是**。

实际输出引用：支持。置信度：high。

原答案重复歌名，未给演唱者；最终明确 Kid Creole and the Coconuts 首次录制，回答了谁唱，引用完整。EM 未识别这一语义修复。

证据定位：R1 `nq_p_001994`：first recorded by his band Kid Creole and the Coconuts

冻结参考（仅作对照）：Kid Creole and the Coconuts

结果 SHA-256：`7a236bf83bd5dbcad8b7fe6a0cd286e527defb51a2d5bad2aa317a74e6efb9d1`；人工状态：pending。

本条与第 23 条审核材料相同；独立保留本批结果指纹。

## 36. NQ / dev50 / -3089533754017582743

问题：when was the last time unc did not make the ncaa tournament

初始答案：2001

最终答案：The state of North Carolina was threatened with a 2018 - 2022 championship venue boycott by the NCAA , due to the HB2 law passed in 2016 .

建议：初始正确 **否** → 最终正确 **否**；Top-5 支持 **否 → 否**。

实际输出引用：不支持。置信度：high。

R3 明确连续参赛延续到 2001，故原 2001 不是缺席年份；最终谈 North Carolina 比赛场地抵制，不是 UNC 未进赛事，事件不匹配。C1 没有完整支持参考 2003 的最近缺席结论。

证据定位：R3 `nq_p_060452`：27 straight appearances in the NCAA tourney；R4 `nq_p_075704`：championship venue boycott

冻结参考（仅作对照）：2003

结果 SHA-256：`d3cb3d2822d82372b820e8aa98963d38d8a09b90012aeb3921040e5e311ac596`；人工状态：pending。

## 37. NQ / dev50 / -5973003791441271262

问题：where does the last name andersen originate from

初始答案：Danish-Norwegian

最终答案：Andersen ( Danish pronunciation : ( ˈɑnɐsn̩ ) ) is a Danish - Norwegian patronymic surname meaning `` son of Anders '' ( itself derived from the Greek name `` Ανδρέας / Andreas '' , cf . English Andrew ) .

建议：初始正确 **是** → 最终正确 **是**；Top-5 支持 **是 → 是**。

实际输出引用：支持。置信度：medium。

按 where originate 的地域/语言来源读法，Danish-Norwegian 已给核心来源；最终补父名姓及 son of Anders 词源，均有据。若要求词源全过程，可提高完整性要求，但不直接判原错误。

证据定位：R1 `nq_p_012031`：is a Danish - Norwegian patronymic surname

冻结参考（仅作对照）：Danish-Norwegian patronymic surname meaning "son of Anders"

结果 SHA-256：`4059b317bd0b75c897c27f17c135c966a0329a045ff017350bb6280b16c69f9b`；人工状态：pending。

## 38. NQ / dev50 / 2393389448844632496

问题：what is loihi and where does it originate from

初始答案：Lōihi Seamount

最终答案：Lōihi Seamount ( also known as Lōʻihi ) is an active submarine volcano about 35 km ( 22 mi ) off the southeast coast of the island of Hawaii .

建议：初始正确 **否** → 最终正确 **是**；Top-5 支持 **否 → 是**。

实际输出引用：支持。置信度：high。

原答案仅重复 Lōihi Seamount 名称；最终补足它是海底活火山及位于夏威夷岛东南近海，满足冻结问题/参考的定义与位置要求。

证据定位：R1 `nq_p_099602`：is an active submarine volcano about 35 km ( 22 mi ) off the southeast coast of the island of Hawaii

冻结参考（仅作对照）：an active submarine volcano about 35 km (22 mi) off the southeast coast of the island of Hawaii

结果 SHA-256：`1f7247f570ef8a83c119d9a4c445f0cf7f30e0226a1834fd1ddf40fda345b1c2`；人工状态：pending。

本条与第 25 条审核材料相同；独立保留本批结果指纹。

## 39. NQ / dev50 / 6781489363263681689

问题：where does the last name rudolph come from

初始答案：Germanic name

最终答案：Rudolph or Rudolf ( French : Rodolphe , Italian , Portuguese and Spanish : Rodolfo ) or Rodolphe is a male first name , and , less commonly , a surname . It is a Germanic name deriving from two stems : Rod or Hrōð , meaning `` fame '' , and olf meaning `` wolf '' ( see also Hroðulf ; cf . Adolf ) .

建议：初始正确 **是** → 最终正确 **是**；Top-5 支持 **是 → 是**。

实际输出引用：支持。置信度：medium。

原 Germanic name 已回答名称来源；最终补 fame/wolf 两词根，R1 支持。属于扩充解释，不计确定错误修复。

证据定位：R1 `nq_p_071825`：It is a Germanic name deriving from two stems

冻结参考（仅作对照）：a Germanic name deriving from two stems: Rod or Hrōð, meaning "fame", and olf meaning "wolf"

结果 SHA-256：`f63ccc82794a7a6b8219a2f135de8de5d1a2c737180647b43faf0caef6966a07`；人工状态：pending。

## 40. NQ / dev50 / 7632941217874741929

问题：what is george harrison song wah wah about

初始答案：frustration with the atmosphere in the group

最终答案：The lyrics reflect his frustration with the atmosphere in the group at that time -- namely , Paul McCartney 's over-assertiveness and criticism of his guitar playing , John Lennon 's lack of engagement with the project and dismissal of Harrison as a songwriter , and Yoko Ono 's constant involvement in the band 's activities .

建议：初始正确 **是** → 最终正确 **是**；Top-5 支持 **是 → 是**。

实际输出引用：结合原段指代后支持。置信度：high。

原答案已正确概括对乐队气氛的不满；最终逐项扩充 McCartney/Lennon/Ono 情境，R1 完整支持。

证据定位：R1 `nq_p_098644`：The lyrics reflect his frustration with the atmosphere in the group at that time

冻结参考（仅作对照）：his frustration with the atmosphere in the group at that time

结果 SHA-256：`ae59ee0305a75755654f660db9637bf809c6fab89003bab945a2dcf685f75a2c`；人工状态：pending。

## 41. NQ / dev50 / 8003762008471695244

问题：the christian church used which holiday to take attention away from samhain

初始答案：Christmas

最终答案：It was a custom of the Pagans to celebrate on the same 25 December the birthday of the Sun , at which they kindled lights in token of festivity . In these solemnities and revelries the Christians also took part . Accordingly when the doctors of the Church perceived that the Christians had a leaning to this festival , they took counsel and resolved that the true Nativity should be solemnised on that day .

建议：初始正确 **否** → 最终正确 **否**；Top-5 支持 **否 → 否**。

实际输出引用：不支持。置信度：high。

原 Christmas 与参考 All Saints 不同；最终引用 12 月 25 日太阳节/圣诞安排，不能解释 10 月底 Samhain 的替代节日。引文真实但事件错误，Top-5 也缺目标替代关系。

证据定位：R1 `nq_p_061336`：celebrated from 31 October to 1 November；R5 `nq_p_050392`：25 December the birthday of the Sun

冻结参考（仅作对照）：All Saints (or All Hallows)

结果 SHA-256：`3b60f197bd75588cdea75702fb04d870872ad92eba5ef2996194bea32dd4149e`；人工状态：pending。

## 42. NQ / dev50 / 970985133701043298

问题：a legislative act passed by congress is an example of

初始答案：statute

最终答案：The act was passed by large majorities in both houses of Congress .

建议：初始正确 **是** → 最终正确 **否**；Top-5 支持 **是 → 否**。

实际输出引用：不支持。置信度：high。

R1 明确 statutes/statutory law 是国会通过的 Acts，原 statute 语义正确。最终只说某法案由两院多数通过，没有回答法律类别；是真实语义误修，但冻结 EM 前后均为 0 将其漏掉。

证据定位：R1 `nq_p_085846`：statutes；R1 `nq_p_085846`：statutory law；R2 `nq_p_083953`：The act was passed by large majorities in both houses of Congress .

冻结参考（仅作对照）：statutory law

结果 SHA-256：`b35ada7901232f1c565b267f02615ce7e52450f52e4a62af9544a19714ae36b7`；人工状态：pending。

## 43. FEVER / reviewed / 1251

问题：Watertown, Massachusetts is in the most populous state in New England.

初始答案：REFUTES

最终答案：REFUTES

建议：初始正确 **否** → 最终正确 **否**；Top-5 支持 **否 → 否**。

实际输出引用：无输出引用。置信度：high。

R1 将 Watertown 定位在 Massachusetts，R2 明确该州是 New England 人口最多的州；两句组合支持声明。前后 REFUTES 均错误，uncertain 回退未修复。

证据定位：R1 `Watertown,_Massachusetts`：The Town of Watertown is a city in Middlesex County , Massachusetts；R2 `Massachusetts`：is the most populous state in the New England region

冻结参考（仅作对照）：SUPPORTS

结果 SHA-256：`af0123ab27460544741271039c5864f4512e7f2338095f78becb1d966cf6304e`；人工状态：pending。

## 44. FEVER / reviewed / 146639

问题：In 2003, Eva Green debuted in film.

初始答案：REFUTES

最终答案：REFUTES

建议：初始正确 **否** → 最终正确 **否**；Top-5 支持 **否 → 否**。

实际输出引用：无输出引用。置信度：high。

R1 明确 Eva Green 于 2003 年在 The Dreamers 首次出演电影；冲突回退仍保留错误 REFUTES。

证据定位：R1 `Eva_Green`：making her film debut in 2003

冻结参考（仅作对照）：SUPPORTS

结果 SHA-256：`4081836353572333091c89611034df505bc768685e7cb73bae241a48c517cf4b`；人工状态：pending。

## 45. FEVER / reviewed / 161905

问题：In 1974 Alan Shepard retired from the US Navy.

初始答案：REFUTES

最终答案：REFUTES

建议：初始正确 **否** → 最终正确 **否**；Top-5 支持 **否 → 否**。

实际输出引用：无输出引用。置信度：high。

R1 明确 1974-08-01 从 US Navy 和 NASA 退休；前后 REFUTES 与直接证据相反。

证据定位：R1 `Alan_Shepard`：retirement from the United States Navy and NASA on August 1 , 1974

冻结参考（仅作对照）：SUPPORTS

结果 SHA-256：`c14d547ee56431896896e0910424ac97ef526a342db50e4f6b048696df6a0300`；人工状态：pending。

## 46. FEVER / reviewed / 17973

问题：Part of Dilwale Dulhania Le Jayenge was shot in August.

初始答案：REFUTES

最终答案：REFUTES

建议：初始正确 **否** → 最终正确 **否**；Top-5 支持 **否 → 否**。

实际输出引用：无输出引用。置信度：high。

拍摄区间为 1994 年 9 月至 1995 年 8 月，支持部分拍摄发生在 August；前后 REFUTES 均未正确使用时间区间。

证据定位：R1 `Dilwale_Dulhania_Le_Jayenge`：from September 1994 to August 1995

冻结参考（仅作对照）：SUPPORTS

结果 SHA-256：`d25de1c0891e5bb27f9cc285348b0f603382fb9c53ebf266ea826771e2c23c9f`；人工状态：pending。

## 47. FEVER / reviewed / 193910

问题：Bea Arthur was born under a different name.

初始答案：REFUTES

最终答案：REFUTES

建议：初始正确 **否** → 最终正确 **否**；Top-5 支持 **否 → 否**。

实际输出引用：不支持。置信度：high。

Bea Arthur 出生名 Bernice Frankel，支持出生姓名不同；前后 REFUTES 均错误。最终引用 Bea Wain，是另一人，模型 kept 也没有得到正确证据。

证据定位：R1 `Bea_Arthur`：born Bernice Frankel；R4 `Bea_Wain`：Bea Wain

冻结参考（仅作对照）：SUPPORTS

结果 SHA-256：`c1c2b49cd61b4aeb04196c553d7a20c271fd7be8390892b5dcea218a6c159c20`；人工状态：pending。

## 48. FEVER / reviewed / 202297

问题：The New Jersey Turnpike has shoulders.

初始答案：SUPPORTS

最终答案：SUPPORTS

建议：初始正确 **是** → 最终正确 **是**；Top-5 支持 **是 → 是**。

实际输出引用：无输出引用。置信度：high。

R1 明确 Turnpike 有 10 ft shoulders；初始及最终 SUPPORTS 均正确、有 Top-5 支持。没有输出引用不等于没有证据，但不能计为已提供有效引用。

证据定位：R1 `New_Jersey_Turnpike`：10 ft shoulders

冻结参考（仅作对照）：SUPPORTS

结果 SHA-256：`8290be1d3fc2a4524ba8d2c1b3cb0eab6147ad83b9b3a37c5489fea6d3bcf317`；人工状态：pending。

## 49. FEVER / reviewed / 32939

问题：Louis Malle produced films.

初始答案：SUPPORTS

最终答案：SUPPORTS

建议：初始正确 **是** → 最终正确 **是**；Top-5 支持 **是 → 是**。

实际输出引用：支持。置信度：high。

R1 明确 Louis Malle produced both French and English language films；保留正确答案，引用直接支持。

证据定位：R1 `Louis_Malle`：he produced both French and English language films

冻结参考（仅作对照）：SUPPORTS

结果 SHA-256：`e1214045e44bc7af42dcb25dd6a702b5cad3814d8fe73e140db1bf66b11d7950`；人工状态：pending。

## 50. FEVER / reviewed / 34695

问题：Johnny Van Zant still performs with Lynyrd Skynyrd.

初始答案：REFUTES

最终答案：REFUTES

建议：初始正确 **否** → 最终正确 **否**；Top-5 支持 **否 → 否**。

实际输出引用：无输出引用。置信度：high。

R2 明确 Johnny Van Zant continues to record and perform with them today；按冻结语料时间，前后 REFUTES 错误。

证据定位：R2 `Johnny_Van_Zant`：continues to record and perform with them today

冻结参考（仅作对照）：SUPPORTS

结果 SHA-256：`89e4fa49473210df2ad65c6c7237d6c8c2a889f69305a729a4a1d9b93e182244`；人工状态：pending。

## 51. FEVER / reviewed / 49650

问题：The State of Palestine denies the Gaza Strip.

初始答案：REFUTES

最终答案：REFUTES

建议：初始正确 **是** → 最终正确 **是**；Top-5 支持 **是 → 是**。

实际输出引用：支持。置信度：medium。

按此前确认的声明读法，denies 与原文 claims Gaza Strip 相反，REFUTES 正确，引用支持。措辞存在歧义，沿用已确认的语料语义，不外推现实政治判断。

证据定位：R1 `State_of_Palestine`：The State of Palestine claims the West Bank

冻结参考（仅作对照）：REFUTES

结果 SHA-256：`c272c7ede495b7a4d52d8796a95154812d0595c2c861c92b749bc007b8827cab`；人工状态：pending。

## 52. FEVER / reviewed / 71180

问题：Babur was a Mughal emperor.

初始答案：REFUTES

最终答案：REFUTES

建议：初始正确 **否** → 最终正确 **否**；Top-5 支持 **否 → 否**。

实际输出引用：无输出引用。置信度：high。

R1 明确 Babur became the first Mughal emperor；冲突回退保留 REFUTES，错误未修复。

证据定位：R1 `Babur`：became the first Mughal emperor

冻结参考（仅作对照）：SUPPORTS

结果 SHA-256：`930296ad246b490877d7f179cac87074bf7ca18d66cf49b4f19f90325e41bc6b`；人工状态：pending。

## 53. FEVER / reviewed / 9735

问题：Appropriation (art) played a significant role in visual arts.

初始答案：SUPPORTS

最终答案：SUPPORTS

建议：初始正确 **是** → 最终正确 **是**；Top-5 支持 **是 → 是**。

实际输出引用：支持。置信度：high。

R1 明确 appropriation 对艺术历史起 significant role，括号范围包括 visual arts；保留 SUPPORTS 正确。

证据定位：R1 `Appropriation_-LRB-art-RRB-`：played a significant role in the history of the arts

冻结参考（仅作对照）：SUPPORTS

结果 SHA-256：`36fbba270c934a978e4c1af410df5b89335b6d045381d842b8e33bc38367a630`；人工状态：pending。

## 54. FEVER / dev50 / 101839

问题：The Mighty Ducks was only produced by a Canadian film producer.

初始答案：REFUTES

最终答案：REFUTES

建议：初始正确 **是** → 最终正确 **是**；Top-5 支持 **待定 → 待定**。

实际输出引用：不支持。置信度：medium。

REFUTES 符合冻结标签。R2 为原电影列出两家制作公司，但未明确制作人国籍，能否充分反驳 only a Canadian producer 留作边界。实际引用 D3 是续集，不能直接作为原电影制作信息。

证据定位：R2 `The_Mighty_Ducks`：It was produced by The Kerner Entertainment Company and Avnet；R3 `D3-COLON-_The_Mighty_Ducks`：D3 : The Mighty Ducks

冻结参考（仅作对照）：REFUTES

结果 SHA-256：`daad7943fb81ed58093af57678aedc8896aa7f615976786b17dcd2b5c380aaa5`；人工状态：pending。

## 55. FEVER / dev50 / 108201

问题：Kate Walsh was on a tour of duty in 1947.

初始答案：REFUTES

最终答案：REFUTES

建议：初始正确 **是** → 最终正确 **是**；Top-5 支持 **否 → 否**。

实际输出引用：不支持。置信度：medium。

REFUTES 符合冻结标签，但 C1 是 Katherine Victoria Walsh、Judy Walsh 等不同人物；引用 Judy Walsh 的职业年份不能判断 Kate Walsh 的 1947 tour of duty。

证据定位：R1 `Katherine_Walsh_-LRB-actress-RRB-`：Katherine Victoria Walsh；R2 `Judy_Walsh`：Judy Walsh is a film actress

冻结参考（仅作对照）：REFUTES

结果 SHA-256：`c528b905840970eebbe05b85cf9b9f8956e8177530ac86ecf3715ec6f9548d40`；人工状态：pending。

## 56. FEVER / dev50 / 119422

问题：The Punisher is exclusively a comic book.

初始答案：REFUTES

最终答案：REFUTES

建议：初始正确 **是** → 最终正确 **是**；Top-5 支持 **是 → 是**。

实际输出引用：支持。置信度：high。

R2 明确 Punisher 角色出现在电视及电影中，反驳 exclusively a comic book；最终引用包含这些跨媒介实例。

证据定位：R2 `Punisher`：In feature films , Dolph Lundgren portrayed the Punisher in 1989

冻结参考（仅作对照）：REFUTES

结果 SHA-256：`ca2c2a73739fca529045415c2513c2d7451b283962749ad1e2da85548f914b46`；人工状态：pending。

## 57. FEVER / dev50 / 121921

问题：The WWE was founded after the creation of the NXT Tag Team Championship.

初始答案：REFUTES

最终答案：REFUTES

建议：初始正确 **是** → 最终正确 **是**；Top-5 支持 **否 → 否**。

实际输出引用：仅部分支持。置信度：medium。

REFUTES 符合冻结标签；R1 能证明 WWE 前身在 1963 年已存在，但 R2 是 NXT Championship，不是 NXT Tag Team Championship，缺少目标赛事创建时间。引用只覆盖比较的一端，不能计完整时间关系支持。

证据定位：R1 `WWE_Championship`：on April 25 , 1963；R2 `NXT_Championship`：The NXT Championship

冻结参考（仅作对照）：REFUTES

结果 SHA-256：`aa6af53c5a2308178ad0b2be4759cf4f07f9cbae686cb4045992d339921c5668`；人工状态：pending。

## 58. FEVER / dev50 / 122252

问题：The Mirny (sloop-of-war) was in expedition.

初始答案：SUPPORTS

最终答案：SUPPORTS

建议：初始正确 **是** → 最终正确 **是**；Top-5 支持 **是 → 是**。

实际输出引用：无输出引用。置信度：high。

Mirny 明确是 1819–1821 First Russian Antarctic Expedition 的第二艘船；保留 SUPPORTS 正确且有直接证据。

证据定位：R1 `Mirny_-LRB-sloop-of-war-RRB-`：the second ship of the First Russian Antarctic Expedition

冻结参考（仅作对照）：SUPPORTS

结果 SHA-256：`66c9ac8c4692feb9c9b4ff112324563bcd754f0173d27765ec6465961596f49e`；人工状态：pending。

## 59. FEVER / dev50 / 127233

问题：Alvin and the Chipmunks is exclusively audio.

初始答案：REFUTES

最终答案：REFUTES

建议：初始正确 **是** → 最终正确 **是**；Top-5 支持 **是 → 是**。

实际输出引用：支持。置信度：high。

C1 有动画电视系列及电影，直接反驳 exclusively audio；最终引用电影及系列关系足够。

证据定位：R1 `Alvin_and_the_Chipmunks_-LRB-1983_TV_series-RRB-`：an American animated television series；R4 `Alvin_and_the_Chipmunks-COLON-_The_Squeakquel`：film is a sequel to the 2007 film Alvin and the Chipmunks

冻结参考（仅作对照）：REFUTES

结果 SHA-256：`52baafa369459cbabe103d3cc1799b0147e3e5c4246a5d133f30e178700fcc6e`；人工状态：pending。

## 60. FEVER / dev50 / 137498

问题：Danger UXB is from 1980.

初始答案：REFUTES

最终答案：REFUTES

建议：初始正确 **是** → 最终正确 **是**；Top-5 支持 **否 → 否**。

实际输出引用：不支持。置信度：medium。

REFUTES 符合冻结标签，但 Top-5 是 UXB/UN 等乐队及公司，没有目标电视剧 Danger UXB 的年份。引用 UN 乐队完全不能支持反驳。

证据定位：R1 `UXB_-LRB-band-RRB-`：UXB is a punk band；R2 `UN_-LRB-band-RRB-`：UN -LRB- also known as United N-generation

冻结参考（仅作对照）：REFUTES

结果 SHA-256：`86fb1a302e2bdd7e7d56dc12aad1c6a2022950f9a42d74b45e0461a2b076aa63`；人工状态：pending。

## 61. FEVER / dev50 / 137715

问题：Winter's Tale was published in 1970.

初始答案：REFUTES

最终答案：REFUTES

建议：初始正确 **是** → 最终正确 **是**；Top-5 支持 **是 → 是**。

实际输出引用：支持。置信度：high。

R3 明确 Winter's Tale 是 1983 年小说，反驳 1970；引用对象和年份均匹配。

证据定位：R3 `Winter's_Tale_-LRB-novel-RRB-`：Winter 's Tale is a 1983 novel by Mark Helprin .

冻结参考（仅作对照）：REFUTES

结果 SHA-256：`d357da0ab3c6e94718d9024a3556dc688fc86df6d29ac083e60dc0f7e4c9c0a8`；人工状态：pending。

## 62. FEVER / dev50 / 142942

问题：Henry Cavill did not star in two Superman films.

初始答案：REFUTES

最终答案：REFUTES

建议：初始正确 **是** → 最终正确 **是**；Top-5 支持 **否 → 否**。

实际输出引用：无输出引用。置信度：medium。

否定 Cavill 主演两部 Superman 的声明应按冻结标签反驳；现有证据明确一部及后续电影存在，却未完整交代他出演第二部，不能视为 C1 已充分支持。

证据定位：R1 `Man_of_Steel_-LRB-film-RRB-`：stars Henry Cavill；R1 `Man_of_Steel_-LRB-film-RRB-`：A follow-up , Batman v Superman

冻结参考（仅作对照）：REFUTES

结果 SHA-256：`e4253396ad63ecdf395457bc144962e0e2c0d15bba8a0360755ffd375d934b00`；人工状态：pending。

## 63. FEVER / dev50 / 151128

问题：Quay premiered in 2017.

初始答案：REFUTES

最终答案：REFUTES

建议：初始正确 **是** → 最终正确 **是**；Top-5 支持 **否 → 否**。

实际输出引用：无输出引用。置信度：medium。

REFUTES 符合冻结标签，但 Top-5 为其他 2015–2017 年影视作品，没有 Quay 的首映信息；保留答案正确性依赖参考，C1 支持不足。

证据定位：R1 `List_of_The_Last_Man_on_Earth_episodes`：The Last Man on Earth；R2 `The_Prison`：The Prison is a 2017

冻结参考（仅作对照）：REFUTES

结果 SHA-256：`d833665280a6a51c2289d2ccc4da42e0cdca2d02cea7ea608d6d95a0b589569f`；人工状态：pending。

## 64. FEVER / dev50 / 159932

问题：Christa McAuliffe taught social studies.

初始答案：REFUTES

最终答案：REFUTES

建议：初始正确 **否** → 最终正确 **否**；Top-5 支持 **否 → 否**。

实际输出引用：不支持。置信度：medium。

按冻结标签 Christa McAuliffe taught social studies 应 SUPPORTS，前后 REFUTES 错误。C1 未检得本人资料；引用 Mary MacKenzie 的声乐教学是错误实体，不能将此案归为证据清楚的 utilisation。

证据定位：R4 `Mary_Mackenzie_-LRB-mezzo-soprano-RRB-`：Mary MacKenzie is an American classical mezzo-soprano and voice teacher

冻结参考（仅作对照）：SUPPORTS

结果 SHA-256：`3e27b85493327e010ffa4a653c5edb866987649e63db3395c801f895efde6135`；人工状态：pending。

## 65. FEVER / dev50 / 16161

问题：Gin derives its main flavour from juniper berries.

初始答案：SUPPORTS

最终答案：SUPPORTS

建议：初始正确 **是** → 最终正确 **是**；Top-5 支持 **是 → 是**。

实际输出引用：支持。置信度：high。

Gin 首句明确主要风味来自 juniper berries；保留正确，引用直接支持。

证据定位：R1 `Gin`：derives its predominant flavour from juniper berries

冻结参考（仅作对照）：SUPPORTS

结果 SHA-256：`3ae3c275737c1ec0f19f1d40e17a6aca15b2dc0ffc873fcd6ec05e30fe78e179`；人工状态：pending。

## 66. FEVER / dev50 / 171625

问题：Dave Gibbons was born at a Planet Fitness.

初始答案：REFUTES

最终答案：REFUTES

建议：初始正确 **是** → 最终正确 **是**；Top-5 支持 **否 → 否**。

实际输出引用：不支持。置信度：medium。

REFUTES 符合冻结标签，但 Top-5 是其他健身人物/机构，未给 Dave Gibbons 出生地点；引用 Marcus Antebi 无关，不能视为有证据核验。

证据定位：R4 `Marcus_Antebi`：Marcus Antebi

冻结参考（仅作对照）：REFUTES

结果 SHA-256：`322b4f8327bac77a6364a382b6c95a39e78d763b7706b0a6600f42e2227922b2`；人工状态：pending。

## 67. FEVER / dev50 / 174027

问题：The Endless River is Pink Floyd's reunion tour.

初始答案：REFUTES

最终答案：REFUTES

建议：初始正确 **是** → 最终正确 **是**；Top-5 支持 **是 → 是**。

实际输出引用：支持。置信度：high。

The Endless River 是 Pink Floyd 的 studio album，并非 reunion tour；实际引用明确 final studio album，反驳成立。

证据定位：R1 `The_Endless_River`：The Endless River is the fifteenth and final studio album；R2 `Pink_Floyd`：The final Pink Floyd studio album , The Endless River

冻结参考（仅作对照）：REFUTES

结果 SHA-256：`55f6596cdfebf9f88f329a93c88af952a974c31208ce46db200d3b9d170267c5`；人工状态：pending。

## 68. FEVER / dev50 / 174599

问题：Artpop had first-week sales of about 258,000 copies in 1999.

初始答案：REFUTES

最终答案：REFUTES

建议：初始正确 **是** → 最终正确 **是**；Top-5 支持 **是 → 是**。

实际输出引用：无输出引用。置信度：high。

R2 明确 Artpop 于 2013 年发行且首周 258,000；声明把时间写成 1999，前后 REFUTES 正确。

证据定位：R2 `Artpop`：released on November 6 , 2013；R2 `Artpop`：first-week sales of 258,000

冻结参考（仅作对照）：REFUTES

结果 SHA-256：`ffaf31a2f186e5a8a33f58c1b558f419f199f4b6bed0d78a76ad8b1c56c747c2`；人工状态：pending。

## 69. FEVER / dev50 / 178904

问题：Dawood Ibrahim was born in the nineteen fifties.

初始答案：REFUTES

最终答案：REFUTES

建议：初始正确 **否** → 最终正确 **否**；Top-5 支持 **否 → 否**。

实际输出引用：无输出引用。置信度：medium。

按冻结标签 Dawood Ibrahim 出生于 1950 年代，REFUTES 错误；C1 是其他名叫 Ibrahim 的人物，不能拿他们的出生年份核实本题。

证据定位：R2 `Ibrahim_Shekarau`：Ibrahim Shekarau；R4 `Ibrahim_Coulibaly`：Ibrahim Coulibaly

冻结参考（仅作对照）：SUPPORTS

结果 SHA-256：`60eef8384a636417a84c0cb63e5bf7652dd64a221213bbf573ca32209a79c470`；人工状态：pending。

## 70. FEVER / dev50 / 181638

问题：Mogadishu is only a business.

初始答案：REFUTES

最终答案：REFUTES

建议：初始正确 **是** → 最终正确 **是**；Top-5 支持 **是 → 是**。

实际输出引用：支持。置信度：high。

Mogadishu 是索马里首都和城市，反驳 only a business；引用港口说明中同样明确 Mogadishu 为 capital of Somalia。

证据定位：R1 `Mogadishu`：the capital and most populous city of Somalia；R4 `Port_of_Mogadishu`：the capital of Somalia

冻结参考（仅作对照）：REFUTES

结果 SHA-256：`553fe560caf67f08a67dea246c79c06b78ee7d51ed483872eefe4df7e70e19e2`；人工状态：pending。

## 71. FEVER / dev50 / 193411

问题：The Eighth Doctor is a BBC science fiction character.

初始答案：SUPPORTS

最终答案：SUPPORTS

建议：初始正确 **是** → 最终正确 **是**；Top-5 支持 **是 → 是**。

实际输出引用：支持。置信度：high。

R3 直接说明 Eighth Doctor 是 BBC science fiction television series Doctor Who 的主角化身，保留 SUPPORTS 正确。

证据定位：R3 `Eighth_Doctor`：the protagonist of the BBC science fiction television series Doctor Who

冻结参考（仅作对照）：SUPPORTS

结果 SHA-256：`c4f27c2a5a06a8bf91afd09de76f400662ef42b5a9697d4909ee24e2345c796c`；人工状态：pending。

## 72. FEVER / dev50 / 211290

问题：The Closer was a cable comedy.

初始答案：REFUTES

最终答案：REFUTES

建议：初始正确 **是** → 最终正确 **是**；Top-5 支持 **否 → 否**。

实际输出引用：不支持。置信度：medium。

REFUTES 符合冻结标签；实际引用的是 1998 年 CBS sitcom The Closer，不能据此核实题目所指 cable 作品的类型。存在同名作品歧义，C1 未提供目标作品的充分信息。

证据定位：R1 `The_Closer_-LRB-1998_TV_series-RRB-`：The Closer is an American television sitcom that aired on CBS for 10 episodes in 1998

冻结参考（仅作对照）：REFUTES

结果 SHA-256：`c058b15a86b88dc54d18757da1d1484e414ef6e999c84ef6243cb2ec194ed195`；人工状态：pending。

## 73. FEVER / dev50 / 217115

问题：Innovation is often also ignored as the application of better solutions that meet existing market needs.

初始答案：REFUTES

最终答案：REFUTES

建议：初始正确 **是** → 最终正确 **是**；Top-5 支持 **是 → 是**。

实际输出引用：无输出引用。置信度：medium。

原文为 often also viewed as 满足需求的更好方案，声明改成 ignored；按该语境和冻结标签，REFUTES 合理。措辞不自然，保留低于直接事实题的置信度。

证据定位：R1 `Innovation`：innovation is often also viewed as the application of better solutions

冻结参考（仅作对照）：REFUTES

结果 SHA-256：`51ef7841a5240e940f3184e6ecf93bd11d7953953894b599df5137dbc1f4c8e4`；人工状态：pending。

## 74. FEVER / dev50 / 219030

问题：Savages was exclusively a German film.

初始答案：REFUTES

最终答案：REFUTES

建议：初始正确 **是** → 最终正确 **是**；Top-5 支持 **否 → 否**。

实际输出引用：不支持。置信度：medium。

REFUTES 符合冻结标签；引用 The Savage Innocents 的英意法合拍信息，不能反驳另一个作品 Savages 的国别。R1 的 Savages 也未明确国籍，C1 支持不足。

证据定位：R1 `Savages_-LRB-1972_film-RRB-`：Savages is a 1972 Merchant Ivory Film；R2 `The_Savage_Innocents`：The Savage Innocents is a 1960 film

冻结参考（仅作对照）：REFUTES

结果 SHA-256：`b1f52465720e209306a99b3de289c4e3656aff4f3527bae7a5ef8939288d1b79`；人工状态：pending。

## 75. FEVER / dev50 / 219125

问题：Valencia has about 800,000 inhabitants in the administrative centre.

初始答案：REFUTES

最终答案：REFUTES

建议：初始正确 **否** → 最终正确 **否**；Top-5 支持 **否 → 否**。

实际输出引用：无输出引用。置信度：medium。

REFUTES 与冻结 SUPPORTS 标签不符。C1 的 829,856 指委内瑞拉 Valencia Municipality，不足以证明西班牙 Valencia administrative centre；保留答案未修复，证据不足也不能转成反驳。

证据定位：R2 `Valencia_Municipality,_Carabobo`：Venezuelan state of Carabobo；R2 `Valencia_Municipality,_Carabobo`：population of 829,856

冻结参考（仅作对照）：SUPPORTS

结果 SHA-256：`b3ee76ebdeb53ec833d7f7365875d2216633d5edf2a9e79c38960065431755b7`；人工状态：pending。

## 76. FEVER / dev50 / 222026

问题：Brubaker is a solely Australian film.

初始答案：REFUTES

最终答案：REFUTES

建议：初始正确 **是** → 最终正确 **是**；Top-5 支持 **是 → 是**。

实际输出引用：支持。置信度：high。

Brubaker 首句明确 American film，反驳 solely Australian；引用实体和国别一致。

证据定位：R2 `Brubaker`：Brubaker is a 1980 American prison drama film

冻结参考（仅作对照）：REFUTES

结果 SHA-256：`61da05159ccf453b61984491137bb98cafb8ba7f970a300a592a7139108b27c1`；人工状态：pending。

## 77. FEVER / dev50 / 222031

问题：Stuart Rosenberg refused to direct Brubaker.

初始答案：REFUTES

最终答案：REFUTES

建议：初始正确 **是** → 最终正确 **是**；Top-5 支持 **是 → 是**。

实际输出引用：支持。置信度：high。

Brubaker 明确 directed by Stuart Rosenberg，反驳其拒绝执导的声明；前后 REFUTES 正确。

证据定位：R1 `Brubaker`：directed by Stuart Rosenberg

冻结参考（仅作对照）：REFUTES

结果 SHA-256：`3cbcff05f71e8165554f748b71509ceecd11c21f64c1e20761dcf5a539b56c21`；人工状态：pending。

## 78. FEVER / dev50 / 222650

问题：A United Kingdom was directed by Amma Asante.

初始答案：SUPPORTS

最终答案：SUPPORTS

建议：初始正确 **是** → 最终正确 **是**；Top-5 支持 **是 → 是**。

实际输出引用：无输出引用。置信度：high。

R1 明确 A United Kingdom directed by Amma Asante，保留 SUPPORTS 正确，虽未输出引用但 C1 足够。

证据定位：R1 `A_United_Kingdom`：directed by Amma Asante

冻结参考（仅作对照）：SUPPORTS

结果 SHA-256：`714a774347c8db0892ff8d5a20d0a9dcf13475efc3b3eeb212b61f1447c8ce49`；人工状态：pending。

## 79. FEVER / dev50 / 222662

问题：A United Kingdom is a biographical romantic drama play.

初始答案：REFUTES

最终答案：REFUTES

建议：初始正确 **是** → 最终正确 **是**；Top-5 支持 **是 → 是**。

实际输出引用：不支持。置信度：high。

R1 明确 A United Kingdom 是 film，足以反驳 play；答案保留正确。但实际引用 Great Britain 这部剧本，实体不同，不能作为该答案的引用依据。

证据定位：R1 `A_United_Kingdom`：A United Kingdom is a 2016 British biographical romantic drama film；R2 `Great_Britain_-LRB-play-RRB-`：Great Britain is a satirical play

冻结参考（仅作对照）：REFUTES

结果 SHA-256：`5e70995ea57352a66097391d279ac83e10f0c4fd6ca5104fa82e742c7cf1cc97`；人工状态：pending。

## 80. FEVER / dev50 / 226143

问题：Richard Dawkins has yet to appear on the radio.

初始答案：REFUTES

最终答案：REFUTES

建议：初始正确 **是** → 最终正确 **是**；Top-5 支持 **是 → 是**。

实际输出引用：支持。置信度：high。

Richard Dawkins 原文明确 regular television, radio and Internet appearances，反驳尚未上广播；引用包含此句。

证据定位：R1 `Richard_Dawkins`：he makes regular television , radio and Internet appearances

冻结参考（仅作对照）：REFUTES

结果 SHA-256：`e1bf5412b2aa2592c1db7390f2316f52789bbe7194090fcf98910fefc1de4cca`；人工状态：pending。

## 81. FEVER / dev50 / 227357

问题：Giada at Home aired on a TV channel.

初始答案：SUPPORTS

最终答案：SUPPORTS

建议：初始正确 **是** → 最终正确 **是**；Top-5 支持 **是 → 是**。

实际输出引用：支持。置信度：high。

Giada at Home 是 Food Network television program，R1 另给首播日期；保留 SUPPORTS，引用主持人条目中的节目/电视信息足够。

证据定位：R2 `Giada_De_Laurentiis`：the current Food Network television program Giada at Home

冻结参考（仅作对照）：SUPPORTS

结果 SHA-256：`a4219344527c76d03a5479bcfa667fda5f4f41fcab7c966d2d536f5f5c2b214f`；人工状态：pending。

## 82. FEVER / dev50 / 227361

问题：Giada at Home aired on the Food Network beginning in 2008.

初始答案：SUPPORTS

最终答案：SUPPORTS

建议：初始正确 **是** → 最终正确 **是**；Top-5 支持 **是 → 是**。

实际输出引用：无输出引用。置信度：high。

R1 明确 first aired on October 18, 2008 on the Food Network，完整支持频道与起始年份。

证据定位：R1 `Giada_at_Home`：It first aired on October 18 , 2008 on the Food Network .

冻结参考（仅作对照）：SUPPORTS

结果 SHA-256：`a552e0b0f0146e2d1b801c1def72b4bccfde4cd4198185f06379c001401e1e15`；人工状态：pending。

## 83. FEVER / dev50 / 28381

问题：Justine Bateman is a poet.

初始答案：REFUTES

最终答案：REFUTES

建议：初始正确 **是** → 最终正确 **是**；Top-5 支持 **否 → 否**。

实际输出引用：不支持。置信度：medium。

REFUTES 符合冻结标签，但 C1 包含 Sarah Holland-Batt、Justine Ettler 等不同人物，无法独立核实 Justine Bateman；实际引用其他诗人不能反驳本题。

证据定位：R1 `Sarah_Holland-Batt`：Sarah Holland-Batt；R4 `Justine_Ettler`：Justine Ettler

冻结参考（仅作对照）：REFUTES

结果 SHA-256：`d9ca7bf08b36f470612740cf6d877e0a28aeefd46f6a60ac68517cd733feee7a`；人工状态：pending。

## 84. FEVER / dev50 / 32338

问题：Dissociative identity disorder is known as dementia.

初始答案：REFUTES

最终答案：REFUTES

建议：初始正确 **是** → 最终正确 **是**；Top-5 支持 **是 → 是**。

实际输出引用：不支持。置信度：medium。

C1 将 DID 定义为多重身份/人格状态障碍，并给 multiple personality disorder 旧称，支持冻结 REFUTES 的类别区分。但实际引用只截取 R3 末尾的 amnesia/fugue/conversion disorder 信息，没有相关 DID 定义；相关页面不等于所选窗口支持答案。

证据定位：R1 `Dissociative_identity_disorder`：also known as multiple personality disorder；R3 `Dissociative_disorder`：Both dissociative amnesia and dissociative fugue

冻结参考（仅作对照）：REFUTES

结果 SHA-256：`cbacf6f329170a7185b434cb6a98d7b3c5d320bdbedc3a4d1cc196d338903aa0`；人工状态：pending。

## 85. FEVER / dev50 / 33498

问题：Ares is not a God.

初始答案：REFUTES

最终答案：REFUTES

建议：初始正确 **是** → 最终正确 **是**；Top-5 支持 **是 → 是**。

实际输出引用：支持。置信度：high。

Ares 明确是 Greek god of war，直接反驳 not a God，引用一致。

证据定位：R2 `Ares_-LRB-disambiguation-RRB-`：Ares is the Greek god of war and violence .

冻结参考（仅作对照）：REFUTES

结果 SHA-256：`b72cd61930f93af6b268b257662bdf516f20261e2e5cfcae6ac488d883b9e4e2`；人工状态：pending。

## 86. FEVER / dev50 / 35822

问题：The Icelandic Coast Guard is responsible for Iceland's maritime search and destroy.

初始答案：REFUTES

最终答案：REFUTES

建议：初始正确 **是** → 最终正确 **是**；Top-5 支持 **是 → 是**。

实际输出引用：不支持。置信度：high。

R1 的职责是 search and rescue，声明替换为 destroy，REFUTES 正确。但实际引用美国 Greenland Patrol，机构/地点均不匹配。

证据定位：R1 `Icelandic_Coast_Guard`：maritime and aeronautical search and rescue；R4 `Greenland_Patrol`：The Greenland Patrol was a United States Coast Guard operation

冻结参考（仅作对照）：REFUTES

结果 SHA-256：`8435c05babe7b9ef98a00e77326c398c9078d0316b54a84517810391ee8d9818`；人工状态：pending。

## 87. FEVER / dev50 / 3668

问题：Andrew Moray refused to cede the territory he gained to John Balliol.

初始答案：REFUTES

最终答案：REFUTES

建议：初始正确 **是** → 最终正确 **是**；Top-5 支持 **是 → 是**。

实际输出引用：无输出引用。置信度：medium。

R1 明确 Moray 夺回地区是 for King John Balliol，支持反驳 refused to cede 的读法；属于关系推断而非逐字否定，保持中置信度。

证据定位：R1 `Andrew_Moray`：successfully regaining control of the area for King John Balliol

冻结参考（仅作对照）：REFUTES

结果 SHA-256：`dc271216978d8cff79f6433817667e3f1ce5aa1ba1066d64f0f2b6fc7a3e329f`；人工状态：pending。

## 88. FEVER / dev50 / 40221

问题：Benzodiazepine is a genre of music.

初始答案：REFUTES

最终答案：REFUTES

建议：初始正确 **是** → 最终正确 **是**；Top-5 支持 **是 → 是**。

实际输出引用：支持。置信度：high。

C1 明确 benzodiazepines 是药物类别，实际引用也将其列为 medication，反驳音乐流派。

证据定位：R1 `Benzodiazepine`：are a class of psychoactive drugs；R5 `Psycholeptic`：Such medications include barbiturates , benzodiazepines

冻结参考（仅作对照）：REFUTES

结果 SHA-256：`77f9a4ce17a8b64cbd6ad2d10887e8c551242a1ee4094c598e44aa8dd80640e2`；人工状态：pending。

## 89. FEVER / dev50 / 40755

问题：Touchscreens are only used in computers.

初始答案：REFUTES

最终答案：REFUTES

建议：初始正确 **是** → 最终正确 **是**；Top-5 支持 **是 → 是**。

实际输出引用：支持。置信度：medium。

按本题 computers 指电脑而非一切计算设备的日常读法，C1 列出手机、游戏机、ATM 等用途，反驳 only。引用包含 phone 及 Nintendo 设备，支持该读法。

证据定位：R1 `Touchscreen`：electronic voting machines , point of sale systems , and smartphones；R3 `Dual-touchscreen`：a computer or phone display setup

冻结参考（仅作对照）：REFUTES

结果 SHA-256：`2c566891877c9570c890f7b7f61cde98ba940f2de41f3488f93376bf5b377d8f`；人工状态：pending。

## 90. FEVER / dev50 / 42213

问题：John Deighton had no choice but to pursue other lines of work.

初始答案：REFUTES

最终答案：REFUTES

建议：初始正确 **否** → 最终正确 **否**；Top-5 支持 **否 → 否**。

实际输出引用：不支持。置信度：medium。

REFUTES 与冻结 SUPPORTS 不符；C1 没有 John Deighton 的经历，实际引用 John Bowdler，不能核实是否被迫转行，也不能视为证据清楚的 utilisation。

证据定位：R2 `John_Bowdler`：John Bowdler

冻结参考（仅作对照）：SUPPORTS

结果 SHA-256：`cd32b393a8a52ee6d297e59bbce8eb329b0f35a87cf7dd6ac75b081b45d1dbea`；人工状态：pending。

## 91. FEVER / dev50 / 42998

问题：Richard Fortus is a Guns N' Roses member.

初始答案：SUPPORTS

最终答案：SUPPORTS

建议：初始正确 **是** → 最终正确 **是**；Top-5 支持 **是 → 是**。

实际输出引用：无输出引用。置信度：high。

R1 明确 Richard Fortus 是 Guns N' Roses 成员，自 2002 年加入；按冻结语料时间保留 SUPPORTS 正确。

证据定位：R1 `Richard_Fortus`：a member of the hard rock band Guns N ' Roses

冻结参考（仅作对照）：SUPPORTS

结果 SHA-256：`6485a842a91df5f370e6b99a8a0233a31cfd604d66abad234c1ec0398f856797`；人工状态：pending。

## 92. FEVER / dev50 / 43698

问题：Younger did not air on TV Land.

初始答案：REFUTES

最终答案：REFUTES

建议：初始正确 **是** → 最终正确 **是**；Top-5 支持 **是 → 是**。

实际输出引用：支持。置信度：high。

R3 明确 Younger 于 2015-03-31 在 TV Land 首播，反驳 did not air；引用直接支持。

证据定位：R3 `List_of_Younger_episodes`：on TV Land

冻结参考（仅作对照）：REFUTES

结果 SHA-256：`019d2787d9c4ff1927c2897824ccbbf41b82c2d3a6e850b071eac8bb9947d0da`；人工状态：pending。

## 93. FEVER / dev50 / 59067

问题：Tatum O'Neal spent her whole life single.

初始答案：REFUTES

最终答案：REFUTES

建议：初始正确 **是** → 最终正确 **是**；Top-5 支持 **否 → 否**。

实际输出引用：不支持。置信度：medium。

REFUTES 符合冻结标签，但检索得到 Shaunie/Shaquille O'Neal，未见 Tatum O'Neal 婚姻资料；引用 Shaunie 的前妻身份不能核实 Tatum。

证据定位：R2 `Shaunie_O'Neal`：Shaunie O'Neal；R2 `Shaunie_O'Neal`：the former wife of NBA center Shaquille O'Neal

冻结参考（仅作对照）：REFUTES

结果 SHA-256：`84682012af6d8370a04087eab17983bec0d0ef31bfce0f261934c3e02d2f9123`；人工状态：pending。

## 94. FEVER / dev50 / 68132

问题：Jens Stoltenberg served as Prime Minister.

初始答案：SUPPORTS

最终答案：SUPPORTS

建议：初始正确 **是** → 最终正确 **是**；Top-5 支持 **是 → 是**。

实际输出引用：支持。置信度：high。

R1 给出 Jens Stoltenberg 两段首相任期，引用 R2 也直接写 Prime Minister Jens Stoltenberg。

证据定位：R2 `Stoltenberg_I_Cabinet`：The Labour Party cabinet was led by Prime Minister Jens Stoltenberg .

冻结参考（仅作对照）：SUPPORTS

结果 SHA-256：`0e538152193f2ebb961b3c8195c7c636f0e281a520dcc02a0e14916eb486c30b`；人工状态：pending。

## 95. FEVER / dev50 / 69422

问题：Rick Ross founded the Maybach Music Group in 2009.

初始答案：SUPPORTS

最终答案：SUPPORTS

建议：初始正确 **是** → 最终正确 **是**；Top-5 支持 **是 → 是**。

实际输出引用：支持。置信度：high。

R2 明确 In 2009, Ross founded Maybach Music Group；年份、人物、机构均一致。

证据定位：R2 `Rick_Ross`：In 2009 , Ross founded the record label Maybach Music Group

冻结参考（仅作对照）：SUPPORTS

结果 SHA-256：`d9ee5f785b2f1f442b8ab15a7b03d7d8868cc8aa1c8e2daf053a265ea97d6db6`；人工状态：pending。

## 96. FEVER / dev50 / 70359

问题：Guns N' Roses started in 1995.

初始答案：REFUTES

最终答案：REFUTES

建议：初始正确 **是** → 最终正确 **是**；Top-5 支持 **是 → 是**。

实际输出引用：支持。置信度：high。

R1/R4 均明确乐队成立于 1985，反驳 1995；引用有效。

证据定位：R4 `Guns_N'_Roses_discography`：formed in Los Angeles , California in 1985

冻结参考（仅作对照）：REFUTES

结果 SHA-256：`8964116c0ac90912e60ad210ab9dfdfbd770acb74af933c4c4fdb16c9324ca8e`；人工状态：pending。

## 97. FEVER / dev50 / 7811

问题：Adobe Photoshop is incapable of being part of the Adobe Systems.

初始答案：REFUTES

最终答案：REFUTES

建议：初始正确 **是** → 最终正确 **是**；Top-5 支持 **是 → 是**。

实际输出引用：支持。置信度：medium。

Photoshop 由 Adobe 开发发行，R1 说明 Adobe Systems 以 Photoshop 著称；按 part of 表示产品归属的读法，REFUTES 有支持。

证据定位：R5 `Photoshop_-LRB-disambiguation-RRB-`：Adobe Photoshop is a graphics editor developed and published by Adobe .

冻结参考（仅作对照）：REFUTES

结果 SHA-256：`a31d253705545c8d2580134d650d1d140029b89688f72dd508bbc089c3bbe695`；人工状态：pending。

## 98. FEVER / dev50 / 8483

问题：The 2003 NCAA Division I Men's Basketball Tournament was single-elimination.

初始答案：SUPPORTS

最终答案：SUPPORTS

建议：初始正确 **是** → 最终正确 **是**；Top-5 支持 **是 → 是**。

实际输出引用：支持。置信度：high。

2003 NCAA 男篮锦标赛首句明确 single-elimination play，保留 SUPPORTS 与引用均正确。

证据定位：R1 `2003_NCAA_Division_I_Men's_Basketball_Tournament`：65 schools playing in single-elimination play

冻结参考（仅作对照）：SUPPORTS

结果 SHA-256：`a6bca4cbdcd20f54c5da47c22b8451b004a2caba44fdaf229decabb93a4f5b6d`；人工状态：pending。

## 99. FEVER / dev50 / 98066

问题：Marco Polo was a person.

初始答案：SUPPORTS

最终答案：SUPPORTS

建议：初始正确 **是** → 最终正确 **是**；Top-5 支持 **是 → 是**。

实际输出引用：支持。置信度：high。

R1 明确 Marco Polo 是 Venetian merchant traveller，并叙述家庭、生卒等人物经历；支持 person。

证据定位：R1 `Marco_Polo`：was a Venetian merchant traveller

冻结参考（仅作对照）：SUPPORTS

结果 SHA-256：`22ec257175a1d662f2d41f68f9297f4d6195df4c5836205e3ba8b280d91cd855`；人工状态：pending。

## 100. FEVER / dev50 / 98655

问题：Faye Resnick is a person.

初始答案：SUPPORTS

最终答案：SUPPORTS

建议：初始正确 **是** → 最终正确 **是**；Top-5 支持 **是 → 是**。

实际输出引用：无输出引用。置信度：high。

R1 明确 Faye Resnick 的出生日期与电视人/作者/设计师身份，支持 person；保留正确。

证据定位：R1 `Faye_Resnick`：is an American television personality , author , and interior designer

冻结参考（仅作对照）：SUPPORTS

结果 SHA-256：`076e90450fac56d12b35a9bbaa9488378d0b6ecec3515fbb31aa7855fa56fc1a`；人工状态：pending。

## 101. NQ / reviewed / -24536083689082297

问题：paleomagnetism is a dating method based on changes in the earth's polarity as demonstrated by

初始答案：geomagnetic reversals

最终答案：geomagnetic reversals

建议：初始正确 **否** → 最终正确 **否**；Top-5 支持 **否 → 否**。

实际输出引用：无输出引用。置信度：high。

沿用你对旧第 17 条的修正：岩层记录提供过去地磁活动和板块位置，火山岩/沉积岩中的地磁反转记录又提供时间尺度；题目 as demonstrated by 需要指出保存记录的岩石材料。geomagnetic reversals 只答现象，忽略记录载体，前后均不完整。

证据定位：R1 `nq_p_046657`：This record provides information on the past behavior of Earth 's magnetic field and the past location of tectonic plates .；R1 `nq_p_046657`：The record of geomagnetic reversals preserved in volcanic and sedimentary rock sequences

冻结参考（仅作对照）：volcanic and sedimentary rock sequences (magnetostratigraphy)

结果 SHA-256：`2516961e113c6b35307801771f8094e8e708d1273cf5d04dcac1eef396e03c0b`；人工状态：pending。

## 102. NQ / reviewed / -2495593332815009891

问题：what does it mean when you roll your thumbs

初始答案：Thumb twiddling

最终答案：Thumb twiddling

建议：初始正确 **否** → 最终正确 **否**；Top-5 支持 **否 → 否**。

实际输出引用：无输出引用。置信度：high。

题问动作的含义，Thumb twiddling 只给名称；R3 明确常表示无用、耗时活动，R1 另提焦虑/习惯。前后均未回答 meaning。

证据定位：R3 `nq_p_049773`：an example of a useless , time - wasting activity

冻结参考（仅作对照）：an example of a useless, time-wasting activity

结果 SHA-256：`b335a91731d1af38e1d8b80a130d583e64bad33b044d7cd236f88e97d137e449`；人工状态：pending。

## 103. NQ / reviewed / -5471777320348063942

问题：when do you get your white coat in pharmacy school

初始答案：at the end of their first academic year

最终答案：at the end of their first academic year

建议：初始正确 **是** → 最终正确 **是**；Top-5 支持 **是 → 是**。

实际输出引用：无输出引用。置信度：medium。

沿用旧第 23 条确认：R5 专指大多数美国药学生在第一学年末领取白袍，优先于 R1 泛指部分医学院的入学前。原答案应加“大多数美国药学生/因校而异”更严谨，但按已确认口径仍可接受；最终未改。

证据定位：R5 `nq_p_032909`：most pharmacy students receive their coats at the end of their first academic year

冻结参考（仅作对照）：before the first year begins

结果 SHA-256：`530fbcb703925d2302c89e99095fa85f73fd2f3709f50e4a4d87c803573ced01`；人工状态：pending。

## 104. NQ / reviewed / -5559132601887673008

问题：who was originally cast to play indiana jones

初始答案：Harrison Ford

最终答案：Harrison Ford

建议：初始正确 **否** → 最终正确 **否**；Top-5 支持 **否 → 否**。

实际输出引用：无输出引用。置信度：high。

R1 区分最初建议 Ford 与最终初选 Tom Selleck；题问 originally cast，原及最终 Harrison Ford 错误，证据清楚却未修复。

证据定位：R1 `nq_p_067908`：finally cast actor Tom Selleck as Indiana Jones

冻结参考（仅作对照）：Tom Selleck

结果 SHA-256：`96f0ba55f445351a266b54fc4db94aa698a3b2e8da03587ff0d061ecf3c56d47`；人工状态：pending。

## 105. NQ / dev50 / -2454338314238497026

问题：what is the name of son of lord krishna

初始答案：Pradyumna

最终答案：Pradyumna

建议：初始正确 **是** → 最终正确 **是**；Top-5 支持 **是 → 是**。

实际输出引用：无输出引用。置信度：high。

R3/R5 明确 Pradyumna 为 Krishna 之子，前后名称正确；无需依赖 R1 的同名国王资料。

证据定位：R3 `nq_p_093564`：Pradyumna was son of Lord Krishna

冻结参考（仅作对照）：Pradyumna

结果 SHA-256：`e937ee60cec5b3b6c93e2a7e40320f16a8edf1023472f3884399844defa855df`；人工状态：pending。

## 106. NQ / dev50 / -2495593332815009891

问题：what does it mean when you roll your thumbs

初始答案：Thumb twiddling

最终答案：Thumb twiddling

建议：初始正确 **否** → 最终正确 **否**；Top-5 支持 **否 → 否**。

实际输出引用：无输出引用。置信度：high。

题问动作的含义，Thumb twiddling 只给名称；R3 明确常表示无用、耗时活动，R1 另提焦虑/习惯。前后均未回答 meaning。

证据定位：R3 `nq_p_049773`：an example of a useless , time - wasting activity

冻结参考（仅作对照）：an example of a useless, time-wasting activity

结果 SHA-256：`01afe0f7188a93b3e134357e9cd2134f737b213c793f352bf90d325a49e3b30c`；人工状态：pending。

本条与第 102 条审核材料相同；独立保留本批结果指纹。

## 107. NQ / dev50 / -3251366420482898925

问题：what is the meaning of auv in cars

初始答案：action utility vehicles

最终答案：action utility vehicles

建议：初始正确 **是** → 最终正确 **是**；Top-5 支持 **是 → 是**。

实际输出引用：无输出引用。置信度：high。

R1 在汽车用语中明确 action utility vehicles (AUVs)，前后正确且有支持。

证据定位：R1 `nq_p_014562`：action utility vehicles ( AUVs )

冻结参考（仅作对照）：action utility vehicles

结果 SHA-256：`7445841e16d44204a5bc2713da367546e0da5a162821f53088a54c9e3b6dd5c3`；人工状态：pending。

## 108. NQ / dev50 / -4111673090986081107

问题：when did hollywood become the centre of the film industry

初始答案：1910

最终答案：1910

建议：初始正确 **否** → 最终正确 **否**；Top-5 支持 **否 → 否**。

实际输出引用：无输出引用。置信度：medium。

1910 在 R1 是并入 Los Angeles 的年份，电影业在 soon thereafter 兴起，不能直接当作成为电影中心的精确年。原答案与冻结 1912 不符；现有 C1 也不足以独立证实 1912。

证据定位：R1 `nq_p_066398`：It was consolidated with the city of Los Angeles in 1910 , and soon thereafter

冻结参考（仅作对照）：1912

结果 SHA-256：`8a0f66b30fbeca0840a0284aeecded4666c487e9c60283107dff6a8516357bbf`；人工状态：pending。

## 109. NQ / dev50 / -4340755100872459608

问题：what does hp mean in war and order

初始答案：Health Points

最终答案：Health Points

建议：初始正确 **是** → 最终正确 **是**；Top-5 支持 **待定 → 待定**。

实际输出引用：无输出引用。置信度：medium。

Health Points 符合参考且 R2 明确游戏通用 HP=hit/health points。但 C1 未点名 War and Order；通用定义能否充分证明该具体游戏的用法，支持性暂留边界。

证据定位：R2 `nq_p_082248`：Health is usually measured in hit points or health points , shortened to HP .

冻结参考（仅作对照）：hit points or health points

结果 SHA-256：`94312e99534a1bacfe8aba38b235e732a587525c0e6767bf990f5882c467ce9e`；人工状态：pending。

## 110. NQ / dev50 / -4393170176714354734

问题：when was the seat moved to the capitol

初始答案：December 19, 1960

最终答案：December 19, 1960

建议：初始正确 **否** → 最终正确 **否**；Top-5 支持 **否 → 否**。

实际输出引用：无输出引用。置信度：high。

R1/R2/R4 明确迁至 Washington/Capitol 是 1800；1960-12-19 是获 National Historic Landmark 的日期，前后都混淆事件。

证据定位：R2 `nq_p_063608`：November 17 , 1800；R5 `nq_p_063617`：On December 19 , 1960 , the Capitol was declared a National Historic Landmark

冻结参考（仅作对照）：1800

结果 SHA-256：`5ec96b3de64da728a3baf4b9521d74356caee5ee543d572709767430cc27333b`；人工状态：pending。

## 111. NQ / dev50 / -4802566247754932209

问题：art of learning from text objects or speech

初始答案：machine learning

最终答案：machine learning

建议：初始正确 **否** → 最终正确 **否**；Top-5 支持 **否 → 否**。

实际输出引用：无输出引用。置信度：medium。

按题目和冻结参考，目标概念是 NLP；C1 说明 machine learning 是应用于语言处理的方法，不能把工具方法直接替换目标领域。题目措辞较省略，但原答案过宽，建议判错并人工复核。

证据定位：R3 `nq_p_046130`：machine learning algorithms have been applied to natural - language processing tasks

冻结参考（仅作对照）：Natural-language processing (NLP)

结果 SHA-256：`5437dda12eee36fb53333db88328e32c7d9e331dcf72d4c2e278b811aece8256`；人工状态：pending。

## 112. NQ / dev50 / -544805225908484128

问题：is greenland part of europe or north america

初始答案：North America

最终答案：North America

建议：初始正确 **是** → 最终正确 **是**；Top-5 支持 **是 → 是**。

实际输出引用：无输出引用。置信度：high。

按地理洲别问题读法，North America 正确且 R1 明确；政治文化与欧洲联系是另一维度，建议最终表述补“地理上”但不判原错。

证据定位：R1 `nq_p_033227`：physiographically a part of the continent of North America

冻结参考（仅作对照）：physiographically a part of the continent of North America

结果 SHA-256：`29d413b1e55e3dd123df85e4f65c056b344094d1a194ba9194cbe552ddfbaeb4`；人工状态：pending。

## 113. NQ / dev50 / -5616003387330422581

问题：how far is the pitchers mound to home plate in little league baseball

初始答案：18.44 m

最终答案：18.44 m

建议：初始正确 **待定** → 最终正确 **待定**；Top-5 支持 **否 → 否**。

实际输出引用：无输出引用。置信度：low。

题目指定 Little League，但 R1 的 18.44m 明确属于 Major League Baseball，R2 是泛化场地描述；Top-5 没有对应年龄组/联盟的投球距离。冻结参考可能沿用了成人场地，前后完整正确性暂无法确认，不能因为数值与参考相同就通过。

证据定位：R1 `nq_p_018535`：In Major League Baseball；R1 `nq_p_018535`：60 feet 6 inches ( 18.44 m )

冻结参考（仅作对照）：exactly 60 feet 6 inches (18.44 m) from the rear point of home plate

结果 SHA-256：`e8d41f5702492985c4ee5ca478db740f619079af0a78165cf99cb2415fe33967`；人工状态：pending。

## 114. NQ / dev50 / -564063817056958231

问题：when does boomer find out she a cylon

初始答案：not explicitly stated in the given context

最终答案：not explicitly stated in the given context

建议：初始正确 **否** → 最终正确 **否**；Top-5 支持 **是 → 是**。

实际输出引用：无输出引用。置信度：high。

C1 描写已接受 Cylon 身份、后续 New Caprica/Downloaded 等情节，没有首次发现的明确集名。前后是有依据的语料不足拒答，未回答题目，不能计为正确答案或错误事实；需检索补证。

证据定位：R1 `nq_p_030724`：She has accepted herself as a Cylon；R5 `nq_p_030723`：In the episode

冻结参考（仅作对照）：Kobol's Last Gleaming

结果 SHA-256：`c110a7a2d7c1b666728c2a4fa184dc90582cd87f9208a9f34cd56e4b78faf70c`；人工状态：pending。

## 115. NQ / dev50 / -6453710656175111438

问题：when was you'll never walk alone first released

初始答案：1945

最终答案：1945

建议：初始正确 **是** → 最终正确 **是**；Top-5 支持 **是 → 是**。

实际输出引用：无输出引用。置信度：high。

R1 将该歌曲源头定位为 1945 年 Carousel，后续 1963 版本是翻唱；按 first released 的作品起源口径，1945 有支持。

证据定位：R1 `nq_p_064580`：from the 1945 Rodgers and Hammerstein musical Carousel

冻结参考（仅作对照）：1945

结果 SHA-256：`1415750c3fd92d1ac230568ee072e4e6fd5597822726d550616ce2d2c2721632`；人工状态：pending。

## 116. NQ / dev50 / -7385216369542237810

问题：who recorded i can't help falling in love with you

初始答案：Elvis Presley

最终答案：Elvis Presley

建议：初始正确 **是** → 最终正确 **是**；Top-5 支持 **是 → 是**。

实际输出引用：无输出引用。置信度：high。

R1 明确 originally recorded by Elvis Presley；前后正确，其他翻唱不影响该答案。

证据定位：R1 `nq_p_011658`：originally recorded by American singer Elvis Presley

冻结参考（仅作对照）：Elvis Presley

结果 SHA-256：`09f23cb961d397c1f50b538a3ab4bd15345d35df710597e88bfbd97123071d51`；人工状态：pending。

## 117. NQ / dev50 / -7843232313103404498

问题：what was the united states foreign policy after ww1 started

初始答案：proclaimed neutrality despite president woodrow wilson's antipathies against germany

最终答案：proclaimed neutrality despite president woodrow wilson's antipathies against germany

建议：初始正确 **是** → 最终正确 **是**；Top-5 支持 **是 → 是**。

实际输出引用：无输出引用。置信度：high。

R2 明确 1914 开战后美国 proclaimed a policy of neutrality，并包含 Wilson 态度的补充；原答案虽长但完全回答政策。

证据定位：R2 `nq_p_090261`：the United States proclaimed a policy of neutrality

冻结参考（仅作对照）：neutrality

结果 SHA-256：`e6752554f733092b7340def88c8cd20614b5e0352cf6a4417270d8217ab5811e`；人工状态：pending。

## 118. NQ / dev50 / 1012541351190124000

问题：what is the new touchback rule in the nfl

初始答案：touchback on the 25-yard line

最终答案：touchback on the 25-yard line

建议：初始正确 **是** → 最终正确 **是**；Top-5 支持 **是 → 是**。

实际输出引用：无输出引用。置信度：medium。

按冻结语料 2016 年 new rule，R1 明确 kickoff touchback 从 20 改至 25 码；原答案给出新位置可接受，但宜加 kickoffs，不能外推所有 touchback 或当前规则。

证据定位：R1 `nq_p_004692`：a touchback on kickoffs at the 25 - yard line instead of the previous 20 - yard line

冻结参考（仅作对照）：a touchback on kickoffs at the 25-yard line instead of the previous 20-yard line

结果 SHA-256：`a90242d03de87b47e441f056bfd981da7888bb2f73e46db9b67782dee110d7f7`；人工状态：pending。

## 119. NQ / dev50 / 165593163908150978

问题：chief election commissioner of india can be removed by whom

初始答案：Parliament

最终答案：Parliament

建议：初始正确 **是** → 最终正确 **是**；Top-5 支持 **是 → 是**。

实际输出引用：无输出引用。置信度：medium。

按冻结语料 R2 的 impeachment by Parliament，原 Parliament 回答了主体；R1 进一步给两院三分之二程序。建议补完整程序，但本题问 whom，短答可接受；此项仅审核存档问答。

证据定位：R2 `nq_p_049359`：The Chief Election Commissioner can be removed only through impeachment by the Parliament .

冻结参考（仅作对照）：two-thirds of the Lok Sabha and the Rajya Sabha

结果 SHA-256：`b69934641330d1624a0933a31de1a225175ca8cd77e3c13ceb320a70ed6e2fad`；人工状态：pending。

## 120. NQ / dev50 / 2178109984531253647

问题：if i die before i wake feed jake original

初始答案：Pirates of the Mississippi

最终答案：Pirates of the Mississippi

建议：初始正确 **是** → 最终正确 **是**；Top-5 支持 **是 → 是**。

实际输出引用：无输出引用。置信度：high。

R2 明确 Feed Jake 由 Pirates of the Mississippi 录制，前后给出乐队名，省略国别/流派不影响答案。

证据定位：R2 `nq_p_054940`：recorded by the American country music band Pirates of the Mississippi

冻结参考（仅作对照）：American country music band Pirates of the Mississippi

结果 SHA-256：`d17f3da2b46e15d4aa7056160d3a267cd3653a2abee3fc2b4a9e2069aa7c5ff3`；人工状态：pending。

## 121. NQ / dev50 / 4305916235259203400

问题：when did the soviet union start and end

初始答案：1922-1991

最终答案：1922-1991

建议：初始正确 **待定** → 最终正确 **待定**；Top-5 支持 **否 → 否**。

实际输出引用：无输出引用。置信度：low。

问题问起止年份，冻结参考 in Eurasia 答非所问。C1 明确 1922 成立，但没有 1991 解体信息；1922–1991 的结束年份无法在本次材料内完整核实，正确性留待确认，支持性不足。

证据定位：R1 `nq_p_048387`：officially established in December 1922

冻结参考（仅作对照）：in Eurasia

结果 SHA-256：`ba5ccc0146d8ddeec3b4cf5e6d1f495e98488e0fa46ec504a03807cc94bf130a`；人工状态：pending。

## 122. NQ / dev50 / 448742752032961281

问题：why god save the queen and not king

初始答案：gender of the reigning monarch

最终答案：gender of the reigning monarch

建议：初始正确 **是** → 最终正确 **是**；Top-5 支持 **是 → 是**。

实际输出引用：无输出引用。置信度：high。

R1/R2 明确 Queen/King 随在位君主性别切换，前后解释正确，不依赖当前君主是谁。

证据定位：R1 `nq_p_086403`：depending on the gender of the reigning monarch

冻结参考（仅作对照）：depending on the gender of the reigning monarch

结果 SHA-256：`e1cd1e3b2f01193a2044e944597f8c787d3e8198067281f7038064324fb12f8d`；人工状态：pending。

## 123. NQ / dev50 / 5332616086855442086

问题：where do green tree frogs go during the day

初始答案：cool, dark, and moist areas

最终答案：cool, dark, and moist areas

建议：初始正确 **是** → 最终正确 **是**；Top-5 支持 **是 → 是**。

实际输出引用：无输出引用。置信度：high。

R1 直接写白天寻找 cool, dark, moist areas；原答案包含核心条件，省略树洞/岩缝举例不影响正确性。

证据定位：R1 `nq_p_072266`：During the day , they find cool , dark , and moist areas

冻结参考（仅作对照）：cool, dark, and moist areas, such as tree holes or rock crevices

结果 SHA-256：`676f8c1e5590859de076aae13935160d3bf256628fdebd4b26e5d06d6e9332d5`；人工状态：pending。

## 124. NQ / dev50 / 5786357257547548864

问题：who died doing the luge in the olympics

初始答案：Nodar Kumaritashvili

最终答案：Nodar Kumaritashvili

建议：初始正确 **是** → 最终正确 **是**；Top-5 支持 **是 → 是**。

实际输出引用：无输出引用。置信度：high。

R3 直接给出 Nodar Kumaritashvili 在 2010 冬奥会训练事故身亡；题目笼统用 in the olympics，姓名正确，可补“赛前训练”使事件更精确。

证据定位：R3 `nq_p_014574`：Nodar Kumaritashvili；R3 `nq_p_014574`：fatal crash during a training run for the 2010 Winter Olympics

冻结参考（仅作对照）：Nodar Kumaritashvili

结果 SHA-256：`2e791543b5b0b18098b29351f426ebb4967fc2dc5a46f2396cbe3a5d62a57e34`；人工状态：pending。

## 125. NQ / dev50 / 6248334821889299282

问题：who sings i just want to use your love tonight

初始答案：The Outfield

最终答案：The Outfield

建议：初始正确 **是** → 最终正确 **是**；Top-5 支持 **待定 → 待定**。

实际输出引用：无输出引用。置信度：medium。

The Outfield 与参考一致，R1 说明其演唱 Your Love；但 C1 没有把题目所引歌词明确对应到该歌，引用片段识别这一步缺直接材料，支持性留边界。

证据定位：R1 `nq_p_036525`：Your Love；R1 `nq_p_036525`：the English rock band the Outfield

冻结参考（仅作对照）：the English rock band the Outfield

结果 SHA-256：`1bd8cb3848b808f52a43c7f90aa31aecbca74861ccb0699cbaeab89ba384fecd`；人工状态：pending。

## 126. NQ / dev50 / 6645756083765359472

问题：what is the name given to the common currency to the european union

初始答案：euro

最终答案：euro

建议：初始正确 **是** → 最终正确 **是**；Top-5 支持 **是 → 是**。

实际输出引用：无输出引用。置信度：high。

R1/R2 明确 euro 是欧盟共同货币项目及多数成员使用货币，回答名称正确；不等于声称所有成员均使用它。

证据定位：R2 `nq_p_073084`：The euro is the result of the European Union 's project for economic and monetary union

冻结参考（仅作对照）：The euro

结果 SHA-256：`a147fd4cb87e806fe6215855d3c06083de1ffed3940b94ac37a10967f92d1566`；人工状态：pending。

## 127. NQ / dev50 / 7804119988370766628

问题：who came up with the theory of relativity

初始答案：Albert Einstein

最终答案：Albert Einstein

建议：初始正确 **是** → 最终正确 **是**；Top-5 支持 **是 → 是**。

实际输出引用：无输出引用。置信度：high。

R1/R3/R5 均将狭义/广义相对论归于 Einstein，前后名称正确，其他研究者贡献不否定该回答。

证据定位：R1 `nq_p_074736`：Albert Einstein published the theory of special relativity in 1905

冻结参考（仅作对照）：Albert Einstein

结果 SHA-256：`f5bb46fba70a715ae36f90ce9d257703bb6ac5c0211b6e7abc0ac5e11308294e`；人工状态：pending。

## 128. NQ / dev50 / 7916191162136893139

问题：when did end of the road come out

初始答案：June 30, 1992

最终答案：June 30, 1992

建议：初始正确 **是** → 最终正确 **是**；Top-5 支持 **是 → 是**。

实际输出引用：无输出引用。置信度：high。

R2 明确 End of the Road 于 1992-06-30 发布，比参考仅写 1992 更具体而且有支持；不能按 EM=0 判错。

证据定位：R2 `nq_p_054567`：was released on June 30 , 1992

冻结参考（仅作对照）：1992

结果 SHA-256：`0bc58c7b08647de7628ddbb9ce21c964915f515360093432b57fc90ae8b4b68b`；人工状态：pending。

