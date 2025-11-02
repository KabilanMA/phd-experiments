{-# LANGUAGE DeriveGeneric #-}

import Data.List
import GHC.Generics (Generic)
import Data.Aeson (ToJSON, FromJSON)

-- sayMe :: (Integral a) => a -> String
sayMe 1 = "One!"
sayMe 2 = "Two!"
sayMe 3 = "Three!"
sayMe x = "Not between 1 and 3"

fact 0 = 1
fact n = n * fact (n-1)

capital :: String -> String
capital "" = "Empty string, whoops!"
capital all@(x:xs) = "The first letter of " ++ all ++ " is " ++ xs

-- maximum' :: (Ord a) => [a] -> a
-- maximum' [] = error "maximum of empty list"
-- maximum' [x] = x
-- maximum' (x:xs)
--     | x > maxTail = x
--     | otherwise = maxTail
--     where maxTail = maximum' xs

replicate' :: (Integral i, Ord i) => i -> a -> [a]
replicate' n x
    | n <= 0 = []
    | otherwise = x : replicate' (n-1) x

take' :: (Num i, Ord i) => i -> [a] -> [a]
take' n _
    | n <= 0 = []
take' _ [] = []
take' n (x:xs) = x : take' (n-1) xs

-- reverse' :: [a] -> [a]
-- reverse' [] = []
-- reverse' (x:xs) = reverse' xs ++ [x]

zip' :: [a] -> [b] -> [(a,b)]
zip' _ [] = []
zip' [] _ = []
zip' (x:xs) (y:ys) = (x,y) : zip' xs ys

elem' :: (Eq a) => a -> [a] -> Bool
elem' a [] = False
elem' a (x:xs)
    | a == x = True
    | otherwise = elem' a xs

quicksort :: (Ord a) => [a] -> [a]
quicksort [] = []
quicksort (x:xs) =
    let smallerSorted = quicksort [a | a <- xs, a <= x]
        biggerSorted = quicksort [a | a <- xs, a > x]
    in smallerSorted ++ [x] ++ biggerSorted

zipWith' :: (a -> b -> c) -> [a] -> [b] -> [c]
zipWith' _ [] _ = []
zipWith' _ _ [] = []
zipWith' f (x:xs) (y:ys) = f x y: zipWith' f xs ys

flip' :: (a -> b -> c) -> (b -> a -> c)
flip' f = g
    where g x y = f y x

map' :: (a -> b) -> [a] -> [b]
map' _ [] = []
map' f (x:xs) = f x : map' f xs

map2 :: Num a => [a] -> [a]
map2 xs = [x+3| x <- xs]

chain :: Integral a => a -> [a]
chain 1 = [1]
chain n
    | even n = n : chain (n `div` 2)
    | odd n = n : chain (3*n+1)

-- numLongChains :: Integral a => a -> a -> [[a]]
numLongChains :: Int
numLongChains = length (filter isLong (map chain [1..10]))
    where isLong xs = length xs > 15

maximum':: (Ord a) => [a] -> a
maximum' = foldr1 (\x acc -> if x > acc then x else acc)

reverse' :: [a] -> [a]
reverse' = foldl (\acc x -> x : acc) []


numUniques :: (Eq a) => [a] -> Int
numUniques = length . nub


phoneBook =
    [("betty","555-2938")
    ,("bonnie","452-2928")
    ,("patsy","493-2928")
    ,("lucille","205-2928")
    ,("wendy","939-8282")
    ,("penny","853-2492")
    ,("penny","679-5313")
    ]

findKey :: (Eq k) => k -> [(k,v)] -> Maybe v
findKey key [] = Nothing
findKey key ((k,v):xs) = if k == key then Just v else findKey key xs

data Shape = Circle Float Float Float | Rectangle Float Float Float Float deriving (Show)
surface :: Shape -> Float
surface (Circle _ _ r) = pi * r ^ 2
surface (Rectangle x1 y1 x2 y2) = abs (x1 - x2) * abs (y1 - y2)

data Pair a b = MkPair a b

data Car a b c = Car { company :: a
                     , model :: b
                     , year :: c 
                     } deriving (Show)


newtype Index = Ix { unIx :: Char }
  deriving (Eq, Show, Generic)