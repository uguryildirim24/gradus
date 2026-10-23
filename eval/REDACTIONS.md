# Redactions in eval/

Written 2026-10-21. Before release, a person's name and references to a personal set of keep/skip labels were taken
out of the eval records. No score, row count, cut-off, rule or decision changed: decisions are credited to "the
author", and a comparison against personal labels was dropped because those labels answer a different question
than the rubric. Several of these files pin others by sha256 (amendment chains, manifests, selection and result
records, data-hash files); every pin of an edited file was updated to its new hash, so each chain verifies as
before. The pin graph (which file pins which) is the same as before the edits.

The write-once result files were not regenerated; only their pin lines were edited by hand.

The manifests now name the blind sheet file in each fresh-test folder `blind_sheet.jsonl` (its content and hash are
unchanged). A checkout that holds `data/` with the sheet under its old name needs it renamed to `blind_sheet.jsonl`
before the hash checks pass.

Deleted from eval/: `diag_cycle3_coops.md` (an analysis against the personal labels; pinned by nothing).

| file | old sha256 | new sha256 | what changed |
|---|---|---|---|
| `eval/cycle3_train_csvs.json` | `615c258edcd45097f172b719fd1a9ec8541e555c9fa9982da2c7a8455da56acf` | `b4ee46c6309277b7127590d8a0db6c0c47545a813522cffc61f52a636599f001` | pins of amendments 5 and 5b updated (no other change) |
| `eval/diag_cycle3_rowread.md` | `5e99bc2e40ae535cc3a8e2e80e3d419417a57762972cfbb5689b5a79a9672277` | `20a89b8e3e37654e6a28e935163e0780598fee6904ab3f4110dc788689c9e3fa` | a person's name replaced by "a blind human-read sheet" (not pinned anywhere) |
| `eval/fresh_test2_manifest.json` | `dd1994a0696cbc1b47b185432f7d8f7983aa73054a8402fd7c414e74fa36041d` | `bd0302f97f9a5f61e4ea84d10560dbd9aee408fc264e9d7024f9473305fa1e3c` | sheet file renamed as above; pin of fresh_test_manifest.json updated |
| `eval/fresh_test3_manifest.json` | `8e8493be091b0ab2207562e0140614b9f7cb3fcb743253c71b09b48fed2009de` | `63bd3cffa2df1dae26978715aaaa5c462c6d7d9fe3baf4bbe6499ff060663f33` | sheet file renamed as above; pins of fresh_test_manifest.json and fresh_test2_manifest.json updated |
| `eval/fresh_test3_result.json` | `ae9ab4932043efd471d3b38d448a0f3533c8474d8da79e2403dc6d0513147426` | `295c66d185d01e89da95578270f13253db1d5fc75255bfd5ec7bb56d1b886bd5` | write-once result, not regenerated: one hand edit, the pin of selection_test3.json; no score changed |
| `eval/fresh_test4_data_hashes.json` | `f251bd7a27ffde780f2e5dd36a50d76b93228e3850afbfb874ddaeb01e18f85f` | `c547fd39198dfde134b3a71c1f086ca959846c7113aebe67a1f0284ef17c2514` | sheet file renamed as above (3 places); pin of fresh_test4_manifest.json updated |
| `eval/fresh_test4_eval_inputs.json` | `d5765d86dce5c27e940408a9bf4c5e0aac252dce68dcce374591f8919c64100a` | `b21fb780b3bb4aedb613370077eb4aa3f555335b3db97fb7e47db3a42e56fa25` | pin of fresh_test4_manifest.json updated (no other change) |
| `eval/fresh_test4_manifest.json` | `eabf019ef0446950050e6f3475da0fba85026cb19701e6a0c29e119a8dc4da26` | `010913c8ce31e180dbfa910e80344256bf53b91975f6fc3a07b4362fd80689f3` | sheet file renamed as above; decided_by credited to the author |
| `eval/fresh_test4_result.json` | `9c03d56929b5bb5e61ae577464825c86c6eaae91a49874228d299d09928402c2` | `6312218c91f7eefca4bdf895e7f5a27ccc2fd21a104c5e8b453b728e41f5f3fb` | write-once result, not regenerated: the pin of selection_cycle3.json updated in its 2 places; no score changed |
| `eval/fresh_test_manifest.json` | `7e2c95d5122f3dab6aa300b24c28f37d2e327a73c6e6899671b6b3a73943fbfd` | `717408f3d184c28cd90daec2677ced08e7ecc54d472f6f837d130c120f70b53d` | sheet file renamed to blind_sheet.jsonl (path and key; the file's hash is unchanged) |
| `eval/protocol_amendment_2.json` | `83e839f744783cc4489192142dde728ff4fcd9b6f1c70ee5be356e05b8bfd4ec` | `6cfaa69e84c139139b28132aaf28d826588473ee25d77c725138ba6b703c74d1` | pin of fresh_test2_manifest.json updated (no other change) |
| `eval/protocol_amendment_3.json` | `40942b97993d375269950e2ec9a01170cf99ba073614b82de3499a36bab880b3` | `b218b1270bfafdbd7a9c0e6ab18e327f4874ea4d257006fc2b8392ec11824615` | a person's name replaced by "the author" (4 places: what, role, never_trained_on, decision.who); pins of amendment 2 and fresh_test3_manifest.json updated |
| `eval/protocol_amendment_3b.json` | `ae393d7d55005185bd2be0a004e96e83868ebd68002524f831fc65deb579d89c` | `a1e3bc3226fd6c48bd5cc7f909285691f32f0db89750c0aff30ee93a83b1a1d5` | a person's name replaced by "the author" (what, decided_by, after_the_result); pins of amendments 2 and 3 and fresh_test3_manifest.json updated |
| `eval/protocol_amendment_4.json` | `1bd8c204a5e54fceeece3fb78512266951e6e6f334ce1b081950dadb2f532875` | `9d3720c0f6a959ff5e71e078cef67c1a685ee38102c86467457faa7d98e770f6` | decided_by credited to the author |
| `eval/protocol_amendment_4b.json` | `8e1a155152a0525defe747993152e65db4b46a054bf4ed6644d1c7cf29cfaddc` | `169df803bce5f72f66a27b933511e6408fbb8c7e0f2503616a12b036df6c8fc1` | decided_by credited to the author; pins of amendments 4 and 5 updated |
| `eval/protocol_amendment_4c.json` | `08b6af9a4d850418febd68291d7fa06bf9fba6d339e04411f3c9db8974bcf43a` | `d344d9becc44a9ff75be4634978d38ab8e5e540c5298a93d13de9ae4eb5b3447` | decided_by credited to the author; pins of amendments 4 and 4b updated |
| `eval/protocol_amendment_5.json` | `49e6ed1afefa286686d917ae886a70520f443ad5e0aa4a6443c7bc5b1a3b7a5b` | `2e0ac6346b39b131e49e67ed71ea9207a7285a205ed3c1dda461001479346aad` | decided_by credited to the author; pins of amendments 2 and 4 updated |
| `eval/protocol_amendment_5b.json` | `b55fbcb862320bbab75ce560498dadca78c1c46a5597553ad66a56fcf9e904e8` | `d959445da0d96038206cbc3a61aa21dd9dae0917403e3adf752e45e7c338157d` | decided_by credited to the author; the clause citing a comparison against a personal set of keep/skip labels (and the deleted doc it linked) replaced by a note that the comparison was dropped, because those labels answer a different question than the rubric; the two in-repo reasons for the 0.60 Keep-recall floor and the floor itself are unchanged; pin of amendment 5 updated |
| `eval/selection_cycle3.json` | `79c372b1cc754b964e358f75c5b1bfc1ca5169c099ad2b4ad1d5c260188e69de` | `17b6bdd3bda9855f9e0e4bb1708b07e8511c608bb5fb8ff9c0b71a66233deab5` | pins of amendments 5 and 5b updated (no other change; the pick and its cut-offs are unchanged) |
| `eval/selection_test3.json` | `eaed7869cf4d6ce734dc4b1ff2373c2979b74d5951e39428824677e892be4bc8` | `9bceb2d07a19b383b37cd0a81836ebe9d308450a09093cb14ada0ab23af7284a` | decided_by credited to the author |

On 2026-10-22, two diagnostic notes that said an analysis came from a "coordinator" were reworded to say "the
author". Neither file is pinned by any other file, and no number changed.

| file | old sha256 | new sha256 | what changed |
|---|---|---|---|
| `eval/diag_cycle3_rowread.md` | `20a89b8e3e37654e6a28e935163e0780598fee6904ab3f4110dc788689c9e3fa` | `090850e11d804da639dfc69429a67e5dbb9a48e754eeae171872c1fe58b26e00` | title: "coordinator row read" became "the author's row read" |
| `eval/diag_keep_levers.md` | `f8efa6c63ad653a8ebf3d722b17a72bb7ef3c916a163783b988deb1a8ed6c05d` | `37b8bcd7be43ae2400c9bf96043ef75a7e173173ba5e629f549601cacf6db9f6` | "a coordinator recompute" became "the author's recompute" |

On 2026-10-23 the amendment 6/6b and fresh test 5 records, which came in after the edits above, got the same
treatment: decisions credited to "the author", the blind sheet named `blind_sheet.jsonl`, no
home-directory paths, and the hand-rated venator set (used only as an overlap reference when drawing test 5's
postings) described without a name. They also pinned several of the files edited on 2026-10-21 by their old hashes;
those pins now carry the new hashes. No score, row count, cut-off, rule or decision changed, and the pin graph is
the same as before the edits. The script that wrote a review sheet outside the repo was not published.

| file | old sha256 | new sha256 | what changed |
|---|---|---|---|
| `eval/cutoffs_B_t1to4.json` | `e26376a27ef385e1d5905a3a28e32ff81c51bd739188cc98ce491a4f64e5ecb1` | `f000820ae730d9e2468911f6ef82f35709f0efbd93b65de7402e2369d7103777` | pins of amendment 6 and of the files redacted earlier (fresh_test4_result.json, amendment 5b, selection_cycle3.json) updated (no other change; the cut-offs are unchanged) |
| `eval/cutoffs_Bcolab_C_t1to4.json` | `86fbfca5e5d6434407fab98e4a6628436dd267bdcc52f79eabd18515df2d43d5` | `cfa751c624a80ba864f2c663d427cc6592ac08cee010cde01169b0b59cdc51e0` | pins of amendment 6b, cutoffs_B_t1to4.json and the files redacted earlier updated (no other change) |
| `eval/cycleC_gates.json` | `632a4460647c4fedc8e30714da913580f48bcf5716efb33345aa05bffca55d01` | `b4af81726ad85245b0d11941fd642a1a2f2791471bbc4931a758be25edd5c526` | gate 2: the review sheet's home-directory path and file name, and the helper script that wrote it (not in the repo), replaced by a description; the gate's waits-on field renamed to waits_on_author (not pinned anywhere) |
| `eval/cycleC_train_csv.json` | `01a0e19762ef59436d53d6ee2c90bfe173735398f0fba7d3c8211824719244f1` | `16c4c757a5f1b68333cc3b47b5139549cbecb65bd5e71d9e9107531d59decaee` | pin of amendment 6b updated (no other change) |
| `eval/fresh_test5_data_hashes.json` | `80a311b741d63b7036a4d10781833a11caaa947899fc6f692ada4ed73aa8b995` | `4f8b3e9dc302888b3bb02af09e604024583e986c5e2a0d03985e9c5681b8697b` | sheet file renamed to blind_sheet.jsonl (path; the file's hash is unchanged); pin of fresh_test5_manifest.json updated |
| `eval/fresh_test5_gold_manifest.json` | `933c1d419e8eb89a91aeae3b23d57bf5c2aa808ebbaa7c740157ee45b6807966` | `6d51f13af27b9ebe1b4e079b21b1f27d522eb369c5107dd093dc2d3c65a8fffb` | pins of fresh_test5_manifest.json and fresh_test5_data_hashes.json updated (no other change) |
| `eval/fresh_test5_manifest.json` | `62bbb4ef1c09231f12a59e0372393dd86c124aa7a41750a5e90dbe0450326069` | `5977d1c55f45547b6bbfc1b10b10e0963ce22bf868252e6a3802a143e63d8c10` | sheet file renamed to blind_sheet.jsonl (path and key); decided_by credited to the author |
| `eval/fresh_test5_result.json` | `d8d4adf15e7d88a91d41c754419f973356071991e3a80f5b79e59e07d6964486` | `039e543c501dee1bcabed65af0f03f92df2597ac1d809d2026a3def63a4ef44a` | write-once result, not regenerated: only its pin lines edited (amendments 6 and 6b, the two cut-off files, selection_cycle3.json); no score changed |
| `eval/protocol_amendment_6.json` | `64b34b94c958791fb0aeb1ca65fc20a0dda0d4a84dd513be45b8404b7f10cf7d` | `aa8ab773af08835cb8778be881ccba1ec125f85cb9283aa812db49b4962dc9e1` | a person's name replaced by "the author" (what, status, decided_by); the hand-rated venator set described without a name (2 places); pins of the files redacted earlier updated |
| `eval/protocol_amendment_6b.json` | `5d276ab43e6cb012ff1fdb2b91d3459079a5254f9e936878f95ca1ecf9262721` | `2e6bca8a4542013be94fbd871a1b091e4b8627663f9ea31e2079bcbf5b569f25` | a person's name replaced by "the author" (status, gate_2, gpu); gate 2's home-directory path replaced by "a review sheet for the author, kept outside the repo"; "the soup's colab_guard.sh" became "the external colab_guard.sh"; pins of amendment 6, fresh_test5_manifest.json, fresh_test5_gold_manifest.json, cutoffs_B_t1to4.json and the files redacted earlier updated |
