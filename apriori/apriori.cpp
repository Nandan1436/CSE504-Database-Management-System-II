#include<bits/stdc++.h>
using namespace std;

vector<set<string>>D;
int min_sup;
vector<map<set<string>,int>>L;

void readInput(){
    ifstream file("input.txt");
    string line;
    while(getline(file,line)){
        if(line.find(',')==string::npos){
            min_sup=stoi(line);
            break;
        }
        set<string>row;
        stringstream ss(line);
        string value;
        getline(ss,value,',');
        while(getline(ss,value,',')){
            row.insert(value);
        }
        D.push_back(row);
    }
}

void find_frequent_1_items(){
    map<set<string>,int>C1;
    for(int i=0;i<D.size();i++){
        for(auto j:D[i]){
            set<string>temp;
            temp.insert(j);
            C1[temp]++;
        }
    }
    for(auto i=C1.begin();i!=C1.end();){
        if(i->second<min_sup){
            i=C1.erase(i);
        }
        else i++;
    }
    L.push_back(C1);
}

bool has_infrequent_subset(set<string>& c, map<set<string>,int>& Lk_minus_1){
    for(auto it=c.begin();it!=c.end();it++){
        set<string>subset=c;
        subset.erase(*it);
        if(Lk_minus_1.find(subset)==Lk_minus_1.end()){
            return true;
        }
    }

    return false;
}

void apriori_gen(map<set<string>,int>& Lk_minus_1, map<set<string>,int>& Ck){
    for(auto l1=Lk_minus_1.begin();l1!=Lk_minus_1.end();l1++){
        auto l2=l1;
        l2++;
        for(;l2!=Lk_minus_1.end();l2++){
            vector<string>v1(l1->first.begin(),l1->first.end());
            vector<string>v2(l2->first.begin(),l2->first.end());

            if(equal(v1.begin(),v1.end()-1,v2.begin(),v2.end()-1) && v1.back()<v2.back()){
                set<string>c=l1->first;
                c.insert(v2.back());
                if(!has_infrequent_subset(c,Lk_minus_1)){
                    Ck[c]=0;
                }
            }
        }
    }
    
}

void count_support(map<set<string>,int>& Ck){
    int count=0;
    for(auto t:D){
        for(auto candidate: Ck){
            if(includes(t.begin(),t.end(),candidate.first.begin(),candidate.first.end())){
                Ck[candidate.first]++;
            }
        }
        
    }
    for(auto it=Ck.begin();it!=Ck.end();){
        if(it->second<min_sup){
            it=Ck.erase(it);
        }
        else{
            it++;
        }
    }
    
}

void find_frequent_k_items(){
    find_frequent_1_items();

    for(int k=1;!L[k-1].empty();k++){
        map<set<string>,int>Ck;
        apriori_gen(L[k-1],Ck);
        count_support(Ck);
        if(!Ck.empty()){
            L.push_back(Ck);
        }
        else break;
    }
}

void printFrequentItemSets(){
    for(int i=0;i<L.size();i++){
        cout<<"L"<<i+1<<":\n";
        for(auto j:L[i]){
            cout<<"{ ";
            for(auto k:j.first){
                cout<<k<<" ";
            }
            cout<<"}->"<<j.second<<"\n";
        }
        cout<<"\n\n";
    }
}

int main(){
    readInput();
    find_frequent_k_items();
    printFrequentItemSets();
    return 0;
}