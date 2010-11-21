(ns bayesian-classifier.core
    (:require [clojure.contrib.duck-streams :as ds]
              [clojure.contrib.pprint :as pp]))


(defn make-classifier [ & klasses]
  (agent {:observations 0
          :classes (reduce
                    (fn [accum k]
                      (assoc accum k { :observations 0 :tokens {} }))
                    {}
                    klasses)}))

(comment
  (def *name-addr-classifier* (make-classifier :name :address))
  (def *health-sick-classifier* (make-classifier :health :sick))
  )

(def *fun-map* (agent  {:observations 0
                        :classes {:name    {:observations 0
                                            :tokens {} }
                                  :address {:observations 0
                                            :tokens {} }
                                  }}))

(defn learn [st token class]
  (let [modify-class
        (fn [token class]
          (let [target-class (get-in st [:classes class])]
            {class {:observations
                    (+ 1 (get target-class :observations))
                    :tokens
                    (assoc (get target-class :tokens) token
                           (+ 1 (get-in target-class [:tokens token] 0)))}}))
        add-token-to-class
        (fn [token class]
          (let [old-classes (reduce (fn [accum k]
                                      (assoc accum k (get-in st [:classes k])))
                                    {}
                                    (filter (fn [some-key] (not (= some-key class)))
                                            (keys (:classes st))))]
            (merge old-classes
                   (modify-class token class))))]
    {:observations (+ 1 (get st :observations))
     :classes (add-token-to-class token class)}))

(comment
  (doseq [a-token  ["paul" "stephanie" "steph" "kyle" "philip" "tom" "sarah" "fred"
                    "albert" "sebastian" "steve" "road" "mary" "vince" "bill" "ben"]]
    (printf "learning about %s\n" a-token )
    (send *name-addr-classifier* learn a-token :name))

  (doseq [a-token  ["street" "road" "lane" "st" "rd" "usa" "philadelphia"]]
    (printf "learning about %s\n" a-token )
    (send *name-addr-classifier* learn a-token :address))

  (dotimes [_ 99]
    (send *health-sick-classifier* learn "+" :sick))
  (send *health-sick-classifier* learn "-" :sick)

  (dotimes [_ 97902]
    (send *health-sick-classifier* learn "-" :health))

  (dotimes [_ 1998]
    (send *health-sick-classifier* learn "+" :health))

  (pp/pprint *health-sick-classifier*)
  (pp/pprint *name-addr-classifier*)

  (p-of-class-given-token *health-sick-classifier* "+")



  )

;;;P(A1), P(A2), ...
(defn p-of-class [st klass]
  (let [total-observations (:observations @st)]
    (if (zero? total-observations)
      0
      (float (/ (get-in @st [:classes klass :observations])
                total-observations)))))

(comment
  (p-of-class *name-addr-classifier* :address)
  (p-of-class *name-addr-classifier* :name)

  )

;;;P(B1|A1), P(B2|A1), ...
(defn p-of-token-given-class [st token class]
  (let [token-count        (get-in @st [:classes class :tokens token] 0)
        class-observations (get-in @st [:classes class :observations]) ]
    (if (zero? class-observations)
      0
      (float (/ token-count class-observations)))))

(comment
  (p-of-token-given-class *name-addr-classifier* "paul" :name)
  (p-of-token-given-class *name-addr-classifier* "paul" :address)
  (p-of-token-given-class *name-addr-classifier* "street" :address)

  )

;;;P(B1|A) = P(B1|A1) + P(B1|A2) + ...
;;;From Graham - ignores prior probablities of class or concept
(defn  p-of-token-given-class-sum [st token]
     (reduce (fn [accum k]
               (+ accum (p-of-token-given-class st token k)))
             0
             (keys (:classes @st))))


(comment
  (keys (:classes @*name-addr-classifier*))
  (p-of-token-given-class *name-addr-classifier* "paul" :address)
  (p-of-token-given-class *name-addr-classifier* "paul" :name)
  (p-of-token-given-class-sum *name-addr-classifier* "paul")

  )

;;;P(A1 && B1 ), P(A1 && B2), ...
(defn p-of-class-and-token [st token klass]
  (* (p-of-class st klass)
     (p-of-token-given-class st token klass)))

(comment
  (p-of-class-and-token *name-addr-classifier* "paul" :name)
  (p-of-class-and-token *name-addr-classifier* "paul" :address)
)

;;;P(B)
(defn total-p-of-token [st token]
  (reduce (fn [accum klass]
            (+ accum
               (p-of-class-and-token st token klass)))
          0
            (keys (:classes @st))))

(comment
  (total-p-of-token *name-addr-classifier* "steph")
  (total-p-of-token *name-addr-classifier* "paul")

  )

;;;P(A|B) = [ P(B|A) * P(A) ]/P(B)
(defn p-of-class-given-token [st token]
  (loop [res {}
         [klass & klasses] (keys (:classes @st))]
    (if-not klass
      res
      (let [numer (p-of-class-and-token st token klass)
            denom (total-p-of-token st token)]
        (printf "%f / %f\n" numer denom)
        (recur (merge res {klass
                           (if (zero? denom)
                             0.0
                             (/ numer denom))})
               klasses)))))

(defn p-of-class-given-token-graham [st token]
  (loop [res {}
         [klass & klasses] (keys (:classes @st))]
    (if-not klass
      res
      (let [numer (p-of-token-given-class st token klass)
            denom (p-of-token-given-class-sum st token)]
        (recur (merge res {klass
                           (if (zero? denom)
                             0.0
                             (/ numer denom))})
               klasses)))))


(comment
  (p-of-class-given-token *name-addr-classifier* "steph")
  (p-of-class-given-token *name-addr-classifier* "paul")
  (p-of-class-given-token *name-addr-classifier* "huge")
  (p-of-class-given-token *name-addr-classifier* "philip")
  (p-of-class-given-token *name-addr-classifier* "road")

  (p-of-class-given-token-graham *name-addr-classifier* "paul")
  (p-of-class-given-token-graham *name-addr-classifier* "road")

  (pp/pprint *name-addr-classifier*)
  (clear-agent-errors *name-addr-classifier*)
  )



