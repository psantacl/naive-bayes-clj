(ns bayesian-classifier.test.core2
  (:require [bayesian-classifier.core2 :as core]
            [clojure.contrib.pprint :as pp] :reload)
  (:use [clojure.test]))



(defn create-classifier [f]
  (def *first-last-name-classifier* (core/make-classifier :first :last))
  (f))

(use-fixtures :each create-classifier)

(deftest test-make-classifier
  (is (= *first-last-name-classifier* {:observations 0
                                      :classes {:first { :observations 0 :tokens {} }
                                                :last { :observations 0 :tokens {} } } })))

(deftest test-learn!-with-bad-class
  (is (thrown? RuntimeException
               (core/learn! *first-last-name-classifier* "fred" :bad-class))))

(deftest test-successful-learn!
  (is (= (core/learn! *first-last-name-classifier* "fred" :first)
         {:observations 1
          :classes {:first { :observations 1 :tokens { "fred" 1 } }
                    :last { :observations 0 :tokens {} } } })))


(deftest test-successful-learn
  (is (= (core/learn *first-last-name-classifier* "germaine" :middle)
         {:observations 1
          :classes {:first  { :observations 0 :tokens {} }
                    :last   { :observations 0 :tokens {} }
                    :middle { :observations 1 :tokens { "germaine" 1 }}
                    }
          })))


(deftest test-p-of-class-given-token
  (doseq [a-token  ["paul" "stephanie" "steph" "kyle" "philip" "tom" "sarah" "fred"
                    "albert" "sebastian" "steve" "smith" "mary" "vince" "bill" "ben"]]
    (def *first-last-name-classifier*
            (core/learn! *first-last-name-classifier* a-token :first)))

  (doseq [a-token ["santa clara" "lesage" "burton" "smith" "smith" "smith" "shires"
                   "mead" "sheppard" "smagghe" "feng"]]
    (def *first-last-name-classifier*
            (core/learn! *first-last-name-classifier* a-token :last)))

  (is (= (core/p-of-class-given-token *first-last-name-classifier* "paul")
           {:first 1.0, :last 0.0}))

  (is (= (core/p-of-class-given-token *first-last-name-classifier* "smith")
         {:first 0.25, :last 0.7499999999999999})))

(deftest p-of-class-given-token-graham
  (doseq [a-token  ["paul" "stephanie" "steph" "kyle" "philip" "tom" "sarah" "fred"
                    "albert" "sebastian" "steve" "smith" "mary" "vince" "bill" "ben"]]
    (def *first-last-name-classifier*
         (core/learn! *first-last-name-classifier* a-token :first)))

  (doseq [a-token ["santa clara" "lesage" "burton" "smith" "smith" "smith" "shires"
                   "mead" "sheppard" "smagghe" "feng"]]
    (def *first-last-name-classifier*
         (core/learn! *first-last-name-classifier* a-token :last)))

  (is (= (core/p-of-class-given-token-graham *first-last-name-classifier* "paul")
         {:first 1.0, :last 0.0}))

  (pp/pprint (core/p-of-class-given-token-graham *first-last-name-classifier* "smith"))

  (is (= (core/p-of-class-given-token-graham *first-last-name-classifier* "smith")
         {:first 0.1864406779661017, :last 0.8135593220338982}))

  )


(comment
  (run-tests 'bayesian-classifier.test.core2)
  (test-p-of-class-given-token)
  )