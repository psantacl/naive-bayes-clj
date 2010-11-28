(ns bayesian-classifier.core
  (:require [clojure.contrib.duck-streams :as ds]
            [clojure.contrib.pprint :as pp]))

(defn make-classifier [& klasses]
  {:observations 0
   :classes (reduce
             (fn [accum k]
               (assoc accum k { :observations 0 :tokens {} }))
             {}
             klasses)})

(defprotocol Classifier
  (get-state [this]  this))

(extend-type java.lang.Object
             Classifier
             (get-state [this] this))

(extend-type clojure.lang.Agent
             Classifier
             (get-state [this] @this))

(extend-type clojure.lang.Atom
             Classifier
             (get-state [this] @this))


(comment
  (def *first-last-name-classifier* (atom (make-classifier :first :last)))
  (def *first-last-name-classifier* (make-classifier :first :last))
  (pp/pprint *first-last-name-classifier*)
  )


(defn base-learn [st token klass]
  (merge  (update-in (update-in st [:classes klass :tokens]
                                (fn [m c] (assoc m token (inc c)))
                                (get-in st [:classes klass :tokens token] 0))
                     [:classes klass]
                     (fn [m c] (assoc m :observations (inc c)))
                     (get-in st [:classes klass :observations]))
          {:observations (inc (get st :observations))}))

(defn learn! [st token klass]
  (if-not (some #(= klass %1) (keys (get st :classes)))
    (throw (RuntimeException. (format "Class/Concept %s not found in classifer"
                                      klass)))
    (base-learn st token klass)))

(defn learn [st token klass]
  (if-not (some #(= klass %1) (keys (get st :classes)))
    (base-learn (update-in st [:classes]
                           (fn [m] (assoc m klass { :observations 0 :tokens {} })))
                token klass)
    (base-learn st token klass)))

(comment
  (doseq [a-token  ["paul" "stephanie" "steph" "kyle" "philip" "tom" "sarah" "fred"
                     "albert" "sebastian" "steve" "smith" "mary" "vince" "bill" "ben"]]
          (reset! *first-last-name-classifier*
                  (learn! @*first-last-name-classifier* a-token :first)))

  (doseq [a-token  ["paul" "stephanie" "steph" "kyle" "philip" "tom" "sarah" "fred"
                    "albert" "sebastian" "steve" "smith" "mary" "vince" "bill" "ben"]]
    (def *first-last-name-classifier*
         (learn! *first-last-name-classifier* a-token :first)))

  (doseq [a-token ["santa clara" "lesage" "burton" "smith" "smith" "smith" "shires"
                    "mead" "sheppard" "smagghe" "feng"]]
          (reset! *first-last-name-classifier*
                  (learn! @*first-last-name-classifier* a-token :last)))

  (doseq [a-token ["santa clara" "lesage" "burton" "smith" "smith" "smith" "shires"
                   "mead" "sheppard" "smagghe" "feng"]]
    (def *first-last-name-classifier*
            (learn! *first-last-name-classifier* a-token :last)))

  (pp/pprint *first-last-name-classifier*)
  )

;;;P(A1), P(A2), ...
(defn p-of-class [st klass]
  (let [total-observations (:observations (get-state st))]
    (if (zero? total-observations)
      0
      (double (/ (get-in (get-state st) [:classes klass :observations])
                total-observations)))))

(comment
  (p-of-class *first-last-name-classifier* :address)
  (p-of-class *first-last-name-classifier* :name)

  )

;;;P(B1|A1), P(B2|A1), ...
(defn p-of-token-given-class [st token class]
  (let [token-count        (get-in (get-state st) [:classes class :tokens token] 0)
        class-observations (get-in (get-state st) [:classes class :observations]) ]
    (if (zero? class-observations)
      0
      (double (/ token-count class-observations)))))

(comment
  (p-of-token-given-class *first-last-name-classifier* "paul" :name)
  (p-of-token-given-class *first-last-name-classifier* "paul" :address)
  (p-of-token-given-class *first-last-name-classifier* "street" :address)

  )

;;;P(B1|A) = P(B1|A1) + P(B1|A2) + ...
;;;From Graham - ignores prior probablities of class or concept
(defn  p-of-token-given-class-sum [st token]
     (reduce (fn [accum k]
               (+ accum (p-of-token-given-class st token k)))
             0
             (keys (:classes (get-state st)))))


(comment
  (keys (:classes *first-last-name-classifier*))
  (p-of-token-given-class *first-last-name-classifier* "paul" :address)
  (p-of-token-given-class *first-last-name-classifier* "paul" :name)
  (p-of-token-given-class-sum *first-last-name-classifier* "paul")

  )

;;;P(A1 && B1 ), P(A1 && B2), ...
(defn p-of-class-and-token [st token klass]
  (* (p-of-class st klass)
     (p-of-token-given-class st token klass)))

(comment
  (p-of-class-and-token *first-last-name-classifier* "paul" :name)
  (p-of-class-and-token *first-last-name-classifier* "paul" :address)
)

;;;P(B)
(defn total-p-of-token [st token]
  (reduce (fn [accum klass]
            (+ accum
               (p-of-class-and-token st token klass)))
          0
            (keys (:classes (get-state st)))))

(comment
  (total-p-of-token *first-last-name-classifier* "steph")
  (total-p-of-token *first-last-name-classifier* "paul")

  )

;;;P(A|B) = [ P(B|A) * P(A) ]/P(B)
(defn p-of-class-given-token [st token]
  (loop [res {}
         [klass & klasses] (keys (:classes (get-state st)))]
    (if-not klass
      res
      (let [numer (p-of-class-and-token st token klass)
            denom (total-p-of-token st token)]
        (recur (merge res {klass
                           (if (zero? denom)
                             0.0
                             (/ numer denom))})
               klasses)))))

(defn p-of-class-given-token-graham [st token]
  (loop [res {}
         [klass & klasses] (keys (:classes (get-state st)))]
    (if-not klass
      res
      (let [numer (p-of-token-given-class st token klass)
            denom (p-of-token-given-class-sum st token)]
        (recur (merge res {klass
                           (if (zero? denom)
                             0.0
                             (/ numer denom))})
               klasses)))))


(defn save-classifier-string [classifier file-name]
  (ds/with-out-writer file-name
    (binding [*print-dup* true]
      (print (get-state classifier)))))


(defn load-classifier-string [file-name]
  (load-file file-name))


(comment
  (save-classifier *first-last-name-classifier*  "chicken.txt")
  (def *persisted-classifier* (load-classifier "chicken.txt"))
  (pp/pprint *persisted-classifier*)
  )

(comment
  (p-of-class-given-token chicken "steph")
  (p-of-class-given-token *first-last-name-classifier* "steph")
  (p-of-class-given-token *first-last-name-classifier* "paul")

  (p-of-class-given-token *first-last-name-classifier* "philip")
  (p-of-class-given-token *first-last-name-classifier* "smith")


  (p-of-class-given-token-graham *first-last-name-classifier* "paul")
  (p-of-class-given-token-graham *first-last-name-classifier* "smith")


  (pp/pprint *first-last-name-classifier*)
  (clear-agent-errors *first-last-name-classifier*)

  )