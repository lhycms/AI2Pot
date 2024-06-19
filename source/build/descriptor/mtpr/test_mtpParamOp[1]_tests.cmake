add_test([=[MtpParamOpTest.forward]=]  /data/home/liuhanyu/hyliu/code/rubbish/test/AI2Pot/source/build/bin/descriptor/mtpr/test_mtpParamOp [==[--gtest_filter=MtpParamOpTest.forward]==] --gtest_also_run_disabled_tests)
set_tests_properties([=[MtpParamOpTest.forward]=]  PROPERTIES WORKING_DIRECTORY /data/home/liuhanyu/hyliu/code/rubbish/test/AI2Pot/source/build/descriptor/mtpr SKIP_REGULAR_EXPRESSION [==[\[  SKIPPED \]]==])
set(  test_mtpParamOp_TESTS MtpParamOpTest.forward)
