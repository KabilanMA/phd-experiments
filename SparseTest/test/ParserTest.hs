{-# LANGUAGE OverloadedStrings #-}

module Main where

import Text.Megaparsec (parseTest)
import DSL.Parser
import System.Directory (listDirectory)
import System.FilePath ((</>))

main :: IO ()
main = do
  files <- listDirectory "tests"
  mapM_ (\f -> do
            putStrLn $ "=== " ++ f ++ " ==="
            src <- readFile ("tests" </> f)
            parseTest parseEinsumStmt src
        ) (filter (".dsl" `isSuffixOf`) files)
